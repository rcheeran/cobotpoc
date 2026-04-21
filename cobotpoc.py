import os
import sys
import glob
import time
import numpy as np
import scipy as sp
import torch
import cv2
import sounddevice as sd
import soundfile as sf
import queue
from contextlib import contextmanager

@contextmanager
def printVerbosely(codeBlockDescription):
    """Print to console before/after block, report outcome and duration"""

    # Print before message
    print(f"{codeBlockDescription}...")
    start = time.time()
    success = True
    
    try:
        yield
    except Exception:
        success = False
        raise

    # Print after message in green or red

    finally:
        elapsed = time.time() - start
        COLORS = {
            "G" : "\033[92m", # Green
            "R" : "\033[91m", # Red
            "W" : "\033[0m", # Default, usually white
        }
        if success:
            print(f"{COLORS['G']}✔ {codeBlockDescription} succeeded in {elapsed:.2f}s{COLORS['W']}")
        else:
            print(f"{COLORS['R']}✗ {codeBlockDescription} failed after {elapsed:.2f}s{COLORS['W']}")

@contextmanager
def suppressStdOut():
    """Suppress stdout and stderr at the OS file descriptor level (catches C++ output from OpenCV, NeMo, etc)"""
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    with open(os.devnull, 'w') as devnull:
        old_stdout = os.dup(stdout_fd)
        old_stderr = os.dup(stderr_fd)
        os.dup2(devnull.fileno(), stdout_fd)
        os.dup2(devnull.fileno(), stderr_fd)
        try:
            yield
        finally:
            os.dup2(old_stdout, stdout_fd)
            os.dup2(old_stderr, stderr_fd)
            os.close(old_stdout)
            os.close(old_stderr)

def listCameras():
    """Return available cameras as dictionaries of index, width, height"""
    allCameras = []
    # Enumerate devices using V4L2 on Linux (anything under /dev/video) and try with OpenCV
    videoDevices = glob.glob('/dev/video*')
    for device in videoDevices:
        index = int(device.replace('/dev/video', ''))
        with suppressStdOut():
            camera = cv2.VideoCapture(index, cv2.CAP_V4L2)
            if camera.isOpened():
                # Verify it's a working camera by reading a frame
                success, frame = camera.read()
                if success and frame is not None:
                    cameraInfo = {
                        "Index" : index,
                        "Width" : int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        "Height": int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    }
                    allCameras.append(cameraInfo)
                camera.release()

    return allCameras or None

def selectBestCamera(allCameras):
    """
    Given list of cameras, select:
        1. The highest resolution, else
        2. The first overall
    """
    
    if not allCameras:
        return None
    
    # 1. Prefer highest resolution
    sortedByRes = sorted(allCameras, key=lambda camera: camera["Width"] * camera["Height"], reverse=True)
    if sortedByRes:
        return sortedByRes[0]
    
    # 2. Fallback: return the first device
    return allCameras[0]

def listMicrophones():
    """Return available microphones as dictionaries of index, name, channels, default_samplerate"""
    devices = sd.query_devices()

    if not devices:
        return None

    allMicrophones = []
    for index, device in enumerate(devices):

        # Skip devices with no input channels
        if device["max_input_channels"] <= 0:
            continue

        # Skip virtual/system devices with suspiciously high channel counts
        if device["max_input_channels"] > 8:
            continue

        # Skip devices with no default sample rate
        if not device.get("default_samplerate"):
            continue

        allMicrophones.append({
            "Name": device["name"],
            "Index": index,
            "Channels": device["max_input_channels"],
            "SampleRate": device["default_samplerate"]
        })

    return allMicrophones or None

def selectBestMicrophone(allMicrophones):
    """
    Given list of microphones, select
        1. The first that includes "USB" in the name, else
        2. The first that has exactly 1 input audio channel, else
        3. The first overall.
    """

    if not allMicrophones:
        return None

    # 1. Prefer USB microphones
    usbMics = [mic for mic in allMicrophones if "usb" in mic["Name"].lower()]
    if usbMics:
        return usbMics[0]

    # 2. Prefer single-channel microphones
    monoMics = [mic for mic in allMicrophones if mic["Channels"] == 1]
    if monoMics:
        return monoMics[0]

    # 3. Fallback: return the first device
    return allMicrophones[0]

def initialize():
    global GPU, CAMERA, VISION_MODEL, VISION_PROCESSOR, MICROPHONE_INDEX, MICROPHONE_SAMPLERATE, SPEECH_MODEL, LLM_TOKENIZER, LLM_MODEL

    # GPU
    with printVerbosely("Check if CUDA is available"):
        hasGPU = torch.cuda.is_available()
        if hasGPU:
            GPU = torch.device("cuda:0")
        else:
            raise RuntimeError("CUDA is not available")
    
    # CAMERA
    with printVerbosely("Prepare camera"):
        allCameras = listCameras()
        if not allCameras:
            raise RuntimeError("No camera found")
        cameraIndex = int(selectBestCamera(allCameras)["Index"])
        CAMERA = cv2.VideoCapture(cameraIndex, cv2.CAP_V4L2)
        if not CAMERA.isOpened():
            raise RuntimeError(f"Could not open camera {cameraIndex}")
        # Prevent camera from buffering many frames ahead
        CAMERA.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # MICROPHONE
    with printVerbosely("Prepare microphone"):
        allMicrophones = listMicrophones()
        if not allMicrophones:
            raise RuntimeError("No microphone found")
        bestMicrophone = selectBestMicrophone(allMicrophones)
        MICROPHONE_INDEX = int(bestMicrophone["Index"])
        MICROPHONE_SAMPLERATE = int(bestMicrophone["SampleRate"])

    # VISION MODEL
    with printVerbosely("Load vision model: grounding-dino-base"):
        # Import HuggingFace "Transformers" library
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection # type: ignore
        # Load Grounding-Dino-Base model (weights about 600MB)
        modelId = "IDEA-Research/grounding-dino-base"
        VISION_PROCESSOR = AutoProcessor.from_pretrained(modelId)
        VISION_MODEL = AutoModelForZeroShotObjectDetection.from_pretrained(modelId).to(GPU)
        VISION_MODEL.eval() # perf optimization since we're only doing inference

    # LANGUAGE MODEL
    with printVerbosely("Load language model: qwen2.5-0.5b-instruct"):
        from transformers import AutoTokenizer, AutoModelForCausalLM # type: ignore
        llmId = "Qwen/Qwen2.5-0.5B-Instruct"
        LLM_TOKENIZER = AutoTokenizer.from_pretrained(llmId)
        LLM_MODEL = AutoModelForCausalLM.from_pretrained(llmId).to("cpu")
        LLM_MODEL.eval()

    # SPEECH MODEL
    with printVerbosely("Load speech model: parakeet-tdt-0.6b-v2"):
        with suppressStdOut():
            import nemo.collections.asr as nemo_asr
            SPEECH_MODEL = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
            SPEECH_MODEL = SPEECH_MODEL.to("cpu")

def makeSpeechCallback(utteranceQueue, minTalkingThreshold=10, maxSilenceThreshold=10, quietThreshold=0.01):
    audioBuffer = []
    silenceCount = 0
    talkingCount = 0

    def callback(indata, frames, callback_time, status):
        """Callback that buffers speech, segments utterances, and pushes segmented audio to a queue."""
        nonlocal audioBuffer, silenceCount, talkingCount

        def isTooQuiet(chunk, threshold=0.03):
            rms = np.sqrt(np.mean(chunk ** 2))
            return rms < threshold

        # Extract, downsample, add to buffer
        audioChunk = indata[:, 0].astype(np.float32)
        audioChunkDownsampled = sp.signal.resample_poly(audioChunk, up=1, down=3)
        audioBuffer.append(audioChunkDownsampled)

        # Use energy threshold to track speaking vs silence in callback thread.
        if isTooQuiet(audioChunkDownsampled, threshold=quietThreshold):
            silenceCount += 1
        else:
            talkingCount += 1
            silenceCount = 0

        # If there hasn't been maxSilenceThreshold of consecutive silence yet, keep listening
        if silenceCount < maxSilenceThreshold:
            return

        # If we've hit maxSilenceThreshold, was there also enough talking? If yes, queue full utterance audio
        if talkingCount >= minTalkingThreshold and len(audioBuffer) >= (minTalkingThreshold + maxSilenceThreshold):
            fullAudio = np.concatenate(audioBuffer, axis=0)
            utteranceQueue.put(fullAudio)

        # Whether or not there was talking, start over
        audioBuffer = []
        silenceCount = 0
        talkingCount = 0

    return callback

def lookForObject(query, threshold=0.1):
    """Capture a single camera frame, detect objects matching query, and display results"""

    # Hue values from 0..180 scale of HSL used by OpenCV
    knownColors = {"red": 0, "yellow": 30, "green": 60, "blue": 120}

    def extractColorFromQuery(query):
        for color in knownColors.keys():
            if color in query.lower():
                return color
        return None

    def computeColorDistance(crop, targetColor):
        """Return absolute difference between average hue of cropped detection and reference color hue"""
        cropHSL = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        cropAvgHue = float(np.mean(cropHSL[:, :, 0]))
        colorDistance = min(abs(cropAvgHue - targetColor), 180 - abs(cropAvgHue - targetColor))
        return colorDistance
    
    # Discard a few frames, in case they were buffered
    CAMERA.grab()
    CAMERA.grab()
    CAMERA.grab()

    # Capture single frame
    success, frame = CAMERA.read()
    if success:
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Object detection
        inputs = VISION_PROCESSOR(images=frameRGB, text=query, return_tensors="pt")
        inputs = {k: (v.to(GPU) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = VISION_MODEL(**inputs)
        
        targetSizes = torch.tensor([frameRGB.shape[:2]], device=GPU)
        results = VISION_PROCESSOR.post_process_grounded_object_detection(outputs, target_sizes=targetSizes, threshold=threshold)[0]
        
        detections = []
        frameH, frameW = frame.shape[:2]
        
        # Process detections
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            score = float(score)
            
            x1, y1, x2, y2 = box.int().tolist()
            
            # Clamp to bounds
            x1, x2 = max(0, x1), min(frameW - 1, x2)
            y1, y2 = max(0, y1), min(frameH - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Ignore detections that are too large
            boxArea = (x2 - x1) * (y2 - y1)
            if boxArea > 0.5 * frameH * frameW:
                continue
            
            # Center 50% crop (for color detection later if needed)
            bw, bh = x2 - x1, y2 - y1
            cx1, cx2 = x1 + bw // 4, x1 + (bw * 3) // 4
            cy1, cy2 = y1 + bh // 4, y1 + (bh * 3) // 4
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            
            detections.append({
                "score": score,
                "label": label,
                "box": (x1, y1, x2, y2),
                "crop": crop
            })
        
        bestDetection = None
        queryColor = extractColorFromQuery(query)
        
        if detections:
            if queryColor is None:
                # Pick highest confidence
                bestDetection = max(detections, key=lambda d: d["score"])
            else:
                # Compute color distance between cropped detection and reference color, and pick closest                
                for det in detections:
                    colorDistance = computeColorDistance(det["crop"], knownColors[queryColor])
                    det["colorDistance"] = colorDistance
                bestDetection = min(detections, key=lambda d: d["colorDistance"])

        # Draw box, label, and confidence on frame for visualization
        drawFrame = frame.copy()
        
        if bestDetection:
            x1, y1, x2, y2 = bestDetection["box"]
            label = bestDetection["label"].strip()
            confidence = bestDetection["score"]
            
            cv2.rectangle(drawFrame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(drawFrame,
                        f"{label} {confidence:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Display result
        cv2.imshow("RESULT", drawFrame)
        cv2.waitKey(1)  # Give 1ms to refresh existing window without blocking

        if bestDetection:
            x1, y1, x2, y2 = bestDetection["box"]
            centerX = (x1 + x2) // 2
            centerY = (y1 + y2) // 2
            return (centerX, centerY)

        return None

    else:
        raise RuntimeError("Failed to read frame from camera")

def transcribeSpeechCommand(speechModel, audio):
    """Transcribe one utterance audio buffer into text."""
    return speechModel.transcribe([audio], verbose=False)[0].text.strip()

def extractObjectFromTranscription(fullText):
    """Use Qwen to extract searchable object(s) from transcribed text, removing fluff."""
    # Prompt Qwen to extract only the object descriptor
    prompt = f"""
        You extract only the searchable object phrase from a spoken request.

        Rules:
        1) If a color is present, you MUST keep the color in the output.
        2) Never drop color words. "blue cube" must stay "blue cube", not "cube".
        3) Remove filler words like "show me", "can you", "please", "I want", "look for".
        4) Return a short noun phrase only, lowercase, with no punctuation.
        5) If no color is provided, return only the object words.

        Examples:
        - "show me the blue cube" -> "blue cube"
        - "can you find a red ball please" -> "red ball"
        - "look for the yellow toy car" -> "yellow toy car"
        - "find the cube" -> "cube"

        Sentence: {fullText}
        Output:"""
    
    # Tokenize and generate
    with torch.inference_mode():
        inputs = LLM_TOKENIZER(prompt, return_tensors="pt")
        # Move inputs to CPU (where LLM is)
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        outputs = LLM_MODEL.generate(**inputs, max_new_tokens=20, temperature=0.3)
        response = LLM_TOKENIZER.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    
    # Clean up response: remove any trailing punctuation, newlines, etc.
    response = response.split('\n')[0].strip()
    return response if response else fullText

def transitionToListeningState():
    global state, query, lookUntil

    state = STATE_LISTENING
    print("Listening...")
    lookUntil = 0.0

    # Clean up
    query = None
    cv2.destroyAllWindows()

def transitionToLookingState():
    global state, query, lookUntil

    state = STATE_LOOKING
    print(f"Looking...")
    lookUntil = time.time() + 10.0

################## MAIN CODE ##################

GPU = None

# Vision
CAMERA = None
VISION_MODEL = None
VISION_PROCESSOR = None

# Language

LLM_TOKENIZER = None
LLM_MODEL = None

# Speech

MICROPHONE_INDEX = None
MICROPHONE_SAMPLERATE = None
SPEECH_MODEL = None

initialize() # Assigns all the global variables above

utteranceQueue = queue.Queue()
SPEECH_CALLBACK = makeSpeechCallback(utteranceQueue)

# State

STATE_LISTENING = "LISTENING"
STATE_LOOKING = "LOOKING"

state = STATE_LISTENING
query = None
lookUntil = 0.0

transitionToListeningState()

# Open microphone stream
with sd.InputStream(device=MICROPHONE_INDEX, channels=1, samplerate=MICROPHONE_SAMPLERATE, blocksize=int(MICROPHONE_SAMPLERATE / 10), dtype="float32", callback=SPEECH_CALLBACK):
    print("Listening... Press Ctrl+C to stop.")
    try:
        while True:
            now = time.time()

            if state == STATE_LISTENING:
                try:
                    # If we have a new utterance, transcribe and switch to looking
                    utteranceAudio = utteranceQueue.get_nowait()
                    fullText = transcribeSpeechCommand(SPEECH_MODEL, utteranceAudio)
                    if fullText:
                        print(f"Heard: '{fullText}'")
                        extractedObject = extractObjectFromTranscription(fullText)
                        print(f"Interpreted: '{extractedObject}'")
                        query = extractedObject
                        transitionToLookingState()
                except queue.Empty:
                    pass

            elif state == STATE_LOOKING:
                # If it's been 10 seconds, go back to listening
                if now >= lookUntil:
                    transitionToListeningState()
                # If we have a query, use the camera to look for it
                elif query:
                    result = lookForObject(query)
                    if result:
                        print(f"Found at (x, y) = {result}")

            sd.sleep(50)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cv2.destroyAllWindows()