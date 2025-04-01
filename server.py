import asyncio
import json
import websockets
import base64
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import logging
import sys
import io
from PIL import Image
import time
import os
from datetime import datetime
import google.generativeai as genai 
from kokoro import KPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = "AIzaSyC0jAjJsgxGUBOvEw8h_L5HlzRVkaS9T-0"  # Replace with your actual key
genai.configure(api_key=GOOGLE_API_KEY)

class AudioSegmentDetector:
    """Detects speech segments and manages processing tasks"""
    
    def __init__(self, 
                 sample_rate=16000,
                 energy_threshold=0.015,
                 silence_duration=0.8,
                 min_speech_duration=0.8,
                 max_speech_duration=15): 
        
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.silence_samples = int(silence_duration * sample_rate)
        self.min_speech_samples = int(min_speech_duration * sample_rate)
        self.max_speech_samples = int(max_speech_duration * sample_rate)
        
        # Internal state
        self.audio_buffer = bytearray()
        self.is_speech_active = False
        self.silence_counter = 0
        self.speech_start_idx = 0
        self.lock = asyncio.Lock()
        self.segment_queue = asyncio.Queue()
        
        # Processing management
        self.current_processing_task = None
        self.processing_lock = asyncio.Lock()
        self.tts_playing = False
        self.tts_lock = asyncio.Lock()
    
    async def cancel_current_processing(self):
        """Cancel the current processing task if exists"""
        async with self.processing_lock:
            if self.current_processing_task and not self.current_processing_task.done():
                self.current_processing_task.cancel()
                try:
                    await self.current_processing_task
                except asyncio.CancelledError:
                    logger.info("Processing task cancelled")
                except Exception as e:
                    logger.error(f"Error cancelling task: {e}")

    async def set_processing_task(self, task):
        """Update the current processing task reference"""
        async with self.processing_lock:
            self.current_processing_task = task

    async def set_tts_playing(self, is_playing):
        """Set TTS playback state"""
        async with self.tts_lock:
            self.tts_playing = is_playing

    async def add_audio(self, audio_bytes):
        """Add audio data to the buffer and check for speech segments"""
        async with self.lock:
            async with self.tts_lock:
                if self.tts_playing:
                    return None
                    
            self.audio_buffer.extend(audio_bytes)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            if len(audio_array) > 0:
                energy = np.sqrt(np.mean(audio_array**2))
                
                if not self.is_speech_active and energy > self.energy_threshold:
                    self.is_speech_active = True
                    self.speech_start_idx = max(0, len(self.audio_buffer) - len(audio_bytes))
                    self.silence_counter = 0
                    logger.info(f"Speech start detected (energy: {energy:.6f})")
                    
                elif self.is_speech_active:
                    if energy > self.energy_threshold:
                        self.silence_counter = 0
                    else:
                        self.silence_counter += len(audio_array)
                        
                        if self.silence_counter >= self.silence_samples:
                            speech_end_idx = len(self.audio_buffer) - self.silence_counter
                            speech_segment = bytes(self.audio_buffer[self.speech_start_idx:speech_end_idx])
                            self.is_speech_active = False
                            self.silence_counter = 0
                            self.audio_buffer = self.audio_buffer[speech_end_idx:]
                            
                            if len(speech_segment) >= self.min_speech_samples * 2:
                                logger.info(f"Speech segment detected: {len(speech_segment)/2/self.sample_rate:.2f}s")
                                await self.segment_queue.put(speech_segment)
                                return speech_segment
                            
                        elif (len(self.audio_buffer) - self.speech_start_idx) > self.max_speech_samples * 2:
                            speech_segment = bytes(self.audio_buffer[self.speech_start_idx:
                                                             self.speech_start_idx + self.max_speech_samples * 2])
                            self.speech_start_idx += self.max_speech_samples * 2
                            logger.info(f"Max duration speech segment: {len(speech_segment)/2/self.sample_rate:.2f}s")
                            await self.segment_queue.put(speech_segment)
                            return speech_segment
            return None
    
    async def get_next_segment(self):
        """Get the next available speech segment"""
        try:
            return await asyncio.wait_for(self.segment_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None

class WhisperTranscriber:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device != "cpu" else torch.float32
        
        model_id = "openai/whisper-small"
        logger.info(f"Loading {model_id}...")
        
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True, 
            use_safetensors=True
        )
        self.model.to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        
        logger.info("Whisper model ready")
        self.transcription_count = 0
    
    async def transcribe(self, audio_bytes, sample_rate=16000):
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio_array) < 1000:
                return ""
            
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.pipe(
                    {"array": audio_array, "sampling_rate": sample_rate},
                    generate_kwargs={
                        "task": "transcribe",
                        "language": "english",
                        "temperature": 0.0
                    }
                )
            )
            
            text = result.get("text", "").strip()
            self.transcription_count += 1
            logger.info(f"Transcription: '{text}'")
            return text
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

class GeminiMultimodalProcessor:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        logger.info("Initializing Gemini...")
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.last_image = None
        self.last_image_timestamp = 0
        self.lock = asyncio.Lock()
        self.generation_count = 0
    
    async def set_image(self, image_data):
        async with self.lock:
            try:
                image = Image.open(io.BytesIO(image_data))
                new_size = (int(image.size[0] * 0.75), int(image.size[1] * 0.75))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                self.last_image = image
                self.last_image_timestamp = time.time()
                return True
            except Exception as e:
                logger.error(f"Image error: {e}")
                return False
    
    async def generate(self, text):
        async with self.lock:
            try:
                if not self.last_image:
                    return f"No image context: {text}"
                
                img_byte_arr = io.BytesIO()
                self.last_image.save(img_byte_arr, format='JPEG')
                img_bytes = img_byte_arr.getvalue()

                prompt = (
                    "You are a helpful assistant providing spoken responses about images. "
                    "Keep answers concise, fluent, and conversational. "
                    "Use natural oral language. Limit to 2-3 short sentences. "
                    "Ask to repeat if unclear.\n\n"
                    f"User input: {text}"
                )
                
                response = await asyncio.to_thread(
                    self.model.generate_content,
                    [prompt, {"mime_type": "image/jpeg", "data": img_bytes}]
                )
                
                output_text = response.text.strip()
                self.generation_count += 1
                logger.info(f"Gemini response ({len(output_text)} chars)")
                return output_text
                
            except Exception as e:
                logger.error(f"Generation error: {e}")
                return f"Error processing: {text}"

class KokoroTTSProcessor:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        logger.info("Initializing Kokoro TTS...")
        try:
            self.pipeline = KPipeline(lang_code='a')
            self.default_voice = 'af_sarah'
            logger.info("Kokoro ready")
            self.synthesis_count = 0
        except Exception as e:
            logger.error(f"TTS init error: {e}")
            self.pipeline = None
    
    async def synthesize_speech(self, text):
        if not text or not self.pipeline:
            return None
        
        try:
            logger.info(f"Synthesizing: '{text[:50]}...'")
            
            audio_segments = []
            generator = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.pipeline(
                    text, 
                    voice=self.default_voice, 
                    speed=1, 
                    split_pattern=r'[.!?。！？]+'
                )
            )
            
            for gs, ps, audio in generator:
                audio_segments.append(audio)
            
            if audio_segments:
                combined_audio = np.concatenate(audio_segments)
                self.synthesis_count += 1
                logger.info(f"Audio synthesized: {len(combined_audio)} samples")
                return combined_audio
            return None
            
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return None

async def process_segment(speech_segment, detector, transcriber, gemini_processor, tts_processor, websocket):
    try:
        # Transcribe
        transcription = await transcriber.transcribe(speech_segment)
        if not transcription:
            return

        # Generate response
        response = await gemini_processor.generate(transcription)
        if not response:
            return

        # Synthesize speech
        await detector.set_tts_playing(True)
        try:
            audio = await tts_processor.synthesize_speech(response)
            if audio is None:
                return

            # Send audio
            audio_bytes = (audio * 32767).astype(np.int16).tobytes()
            base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
            await websocket.send(json.dumps({"audio": base64_audio}))

            # Wait for playback with cancellation support
            total_duration = len(audio) / 24000
            interval = 0.5
            intervals = int(total_duration / interval)

            for _ in range(intervals):
                await asyncio.sleep(interval)
                await websocket.ping()

            remaining = total_duration - (intervals * interval)
            if remaining > 0:
                await asyncio.sleep(remaining)
                await websocket.ping()

        except asyncio.CancelledError:
            logger.info("Playback interrupted")
            raise
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed during playback")
        finally:
            await detector.set_tts_playing(False)

    except asyncio.CancelledError:
        logger.info("Processing cancelled")
        await detector.set_tts_playing(False)
        raise
    except Exception as e:
        logger.error(f"Processing error: {e}")
        await detector.set_tts_playing(False)

async def detect_speech_segments(detector, transcriber, gemini_processor, tts_processor, websocket):
    while True:
        try:
            speech_segment = await detector.get_next_segment()
            if speech_segment:
                # Cancel current processing
                await detector.cancel_current_processing()

                # Start new processing task
                processing_task = asyncio.create_task(
                    process_segment(speech_segment, detector, transcriber, 
                                  gemini_processor, tts_processor, websocket)
                )
                await detector.set_processing_task(processing_task)

            await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            logger.info("Detection cancelled")
            raise
        except Exception as e:
            logger.error(f"Detection error: {e}")

async def handle_client(websocket):
    try:
        await websocket.recv()
        logger.info("Client connected")
        
        detector = AudioSegmentDetector()
        transcriber = WhisperTranscriber.get_instance()
        gemini_processor = GeminiMultimodalProcessor.get_instance()
        tts_processor = KokoroTTSProcessor.get_instance()
        
        async def send_keepalive():
            while True:
                try:
                    await websocket.ping()
                    await asyncio.sleep(10)
                except Exception:
                    break
        
        async def receive_audio_and_images():
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if "realtime_input" in data:
                        for chunk in data["realtime_input"]["media_chunks"]:
                            if chunk["mime_type"] == "audio/pcm":
                                audio_data = base64.b64decode(chunk["data"])
                                await detector.add_audio(audio_data)
                            elif chunk["mime_type"] == "image/jpeg":
                                image_data = base64.b64decode(chunk["data"])
                                await gemini_processor.set_image(image_data)
                    
                    if "image" in data:
                        image_data = base64.b64decode(data["image"])
                        await gemini_processor.set_image(image_data)
                        
                except Exception as e:
                    logger.error(f"Receive error: {e}")

        detect_task = asyncio.create_task(
            detect_speech_segments(detector, transcriber, gemini_processor, tts_processor, websocket)
        )

        await asyncio.gather(
            receive_audio_and_images(),
            detect_task,
            send_keepalive(),
            return_exceptions=True
        )
        
    except websockets.exceptions.ConnectionClosed:
        logger.info("Connection closed")
    except Exception as e:
        logger.error(f"Session error: {e}")
    finally:
        await detector.set_tts_playing(False)
        await detector.cancel_current_processing()

async def main():
    try:
        # Pre-initialize models
        WhisperTranscriber.get_instance()
        GeminiMultimodalProcessor.get_instance()
        KokoroTTSProcessor.get_instance()
        
        logger.info("Starting server on 0.0.0.0:9073")
        async with websockets.serve(
            handle_client, 
            "0.0.0.0", 
            9073,
            ping_interval=20,
            ping_timeout=60,
            close_timeout=10
        ):
            logger.info("Server running")
            await asyncio.Future()
    except Exception as e:
        logger.error(f"Server error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
