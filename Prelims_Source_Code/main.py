
import json
import logging
from dotenv import load_dotenv
import os
from google import genai
from google.genai import types
from faster_whisper import WhisperModel

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

####### logging.CRITICAL for only critical errors, logging.INFO for all info #######
logger.setLevel(logging.INFO) 
# global variables
whispering_shadows = ["atlas_2025", "eos_2023", "hyperion_2022", "oceanus_2022",
                        "rhea_2024", "selene_2024", "titan_2023"]
transcript_file_name_input = "transcribed.txt"
json_file_name_input = "PrelimsSubmission.json"
transcript_file_name_default = "transcribed.txt"
json_file_name_default = "PrelimsSubmission.json"

class TruthWeaverAI:
    
    def __init__(self, transcript_file_name=transcript_file_name_default, json_file_name=json_file_name_default):
        # Initialize faster-whisper model, Google Gemini client, and create transcript file, json file
        logger.info("Initializing Truth Weaver...")

        self.transcript_file_name = transcript_file_name
        self.json_file_name = json_file_name

        with open(self.transcript_file_name, 'w', encoding='utf-8') as f:
            f.write("")  # Clear previous transcript/create file
        
        with open(self.json_file_name, 'w', encoding='utf-8') as f:
            f.write("")  # Clear previous analysis/create file

        self.whisper_model = WhisperModel("base",device="cpu", compute_type="int8")
        self.gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        

        logger.info(f"Truth Weaver initialized")

    def transcribe_audio(self, audio_file_path: str) -> str:
        # speech to text using faster-whisper
        logger.info(f"Transcribing audio: {audio_file_path}")
        
        try:
            segments, info = self.whisper_model.transcribe(audio_file_path)
            transcript_write = ""
            transcript_analyse = ""
            for segment in segments:
                segment_text = segment.text.strip()
                transcript_analyse += segment_text + " "
            # making sure only alphabets and space is there (lowercase)
                segment_text = ''.join(c for c in segment_text if c.isalpha() or c.isspace()).lower()
                transcript_write += segment_text + " "
            
            logger.info(f"{audio_file_path}: {transcript_analyse.strip()}")
            return transcript_write.strip(), transcript_analyse.strip()
            
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            return ""
    
    def transcribe_shadow_sessions(self, whispering_shadow_name: str) -> str:
        """
        Process all sessions for a given whispering shadow and save full transcript
        Assuming that audio files are stored in audio/ directory
        """
        logger.info(f"Transcribing audio sessions of {whispering_shadow_name}")
        ####################################################
        ########### update audio file paths here ###########
        ####################################################
        audio_files = [
            f"audio/{whispering_shadow_name}_1.mp3",
            f"audio/{whispering_shadow_name}_2.mp3",
            f"audio/{whispering_shadow_name}_3.mp3",
            f"audio/{whispering_shadow_name}_4.mp3",
            f"audio/{whispering_shadow_name}_5.mp3"
        ]
        
        full_transcript = ""
        full_transcript_analyse = ""
        
        for i, audio_file in enumerate(audio_files, 1):
            # Transcribe and add to full transcript
            session_transcript, session_transcript_analyse = self.transcribe_audio(audio_file)
            full_transcript += f"{whispering_shadow_name}_{i}.mp3: {session_transcript}\n"
            full_transcript_analyse += session_transcript_analyse + " "

        
        logger.info(f"{whispering_shadow_name}: {full_transcript_analyse}")
        return full_transcript, full_transcript_analyse
    
    def analyse_shadow(self, whispering_shadow_name: str, whispering_shadow_transcript: str):
        # Analyze a single whispering shadow's transcript using Gemini and create json output
        with open("system_prompt.txt", 'r', encoding='utf-8') as sp:
            system_instruction = sp.read()
        try:
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction),
                contents= whispering_shadow_transcript
            )
            first_curly = response.text.find('{') #for getting rid of extra text in front or end
            last_curly = response.text.rfind('}')
            response_json = {}
            response_json['shadow_id'] = f"{whispering_shadow_name}"
            # add response to response_json already containing shadow_id
            gemini_json = json.loads(response.text[first_curly:last_curly+1])
            response_json.update(gemini_json)
            # convert back to json string with indentation
            logger.info(f"Analysis of {whispering_shadow_name}: {json.dumps(response_json, indent=4)}")
            return response_json
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {}

def main():
    # Initialize Truth Weaver
    truth_weaver = TruthWeaverAI(transcript_file_name_input, json_file_name_input)
    
    
    try:
        # for each whispersing shadow, transcribe all sessions and also analyse them
        # save transcript to transcribed.txt and analysis to PrelimsSubmission.json
        tf = open(truth_weaver.transcript_file_name, 'a', encoding='utf-8')
        jf = open(truth_weaver.json_file_name, 'a', encoding='utf-8')
        analysis_list = []

        for i in range(len(whispering_shadows)):
        # transcribing all sessions of the subject
            print(f"Transcribing {whispering_shadows[i]}")
            full_transcript, full_transcript_analyse = truth_weaver.transcribe_shadow_sessions(whispering_shadows[i])
            tf.write(full_transcript)
        # analysing the subject
            print(f"Analysing {whispering_shadows[i]}")
            analysis_json = truth_weaver.analyse_shadow(whispering_shadows[i], full_transcript_analyse)
            analysis_list.append(analysis_json)
        
        jf.write(json.dumps(analysis_list, indent=4))
        tf.close()
        jf.close()


    except Exception as e:
        logger.error(f"Transcription/Analysis failed: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()