
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "GPT_SoVITS"))

import soundfile as sf
import numpy as np
from GPT_SoVITS.TTS_infer_pack.TTS import TTS_Config, TTS

T2S_CKPT = "models/Hutao_GPT.ckpt"
VITS_PTH = "models/Hutao_Sovits.pth"
BERT_PATH = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
HUBERT_PATH = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"

TEXT_LANG = "ko"
REF_AUDIO = "example.wav" 
PROMPT_TEXT = "그나저나, 클로린드가 옛날에 푸리나의 부하였나봐?"
PROMPT_LANG = "ko"

def main():
    # 1. 설정 객체 생성
    custom_cfg = {
        "device": "cuda", 
        "is_half": False,
        "version": "v2",
        "t2s_weights_path": T2S_CKPT,
        "vits_weights_path": VITS_PTH,
        "bert_base_path": BERT_PATH,
        "cnhuhbert_base_path": HUBERT_PATH,
    }
    tts_config = TTS_Config(custom_cfg)
    print(tts_config)
    # 2. TTS 초기화
    tts = TTS(tts_config)

    # 3. 요청 구성

    # 4. 추론 실행
    for i in range(10):
        user_input = input("입력:")
        req = {
            "text": user_input,
            "text_lang": TEXT_LANG,
            "ref_audio_path": REF_AUDIO,
            "aux_ref_audio_paths": ["example_wavs/0_audio.wav","example_wavs/1_audio.wav","example_wavs/2_audio.wav","example_wavs/3_audio.wav","example_wavs/4_audio.wav","example_wavs/5_audio.wav","example_wavs/6_audio.wav","example_wavs/7_audio.wav","example_wavs/8_audio.wav","example_wavs/9_audio.wav","example_wavs/10_audio.wav"],
            "prompt_lang": PROMPT_LANG,
            "top_k": 5,
            "top_p": 1.0,
            "temperature": 1.0,
            "text_split_method": "cut0",
            "batch_size": 1,
            "batch_threshold": 0.75,
            "split_bucket": True,
            "speed_factor": 1.0,
            "streaming_mode": False,
        }
        generator = tts.run(req)
        sr, audio = next(generator)

        # 5. 저장
        file_name = "output"+str(i)+".wav"
        audio_f32 = audio.astype(np.float32) / 32768.0
        sf.write("output"+str(i)+".wav", audio_f32, sr)
        print(f"[✔] 음성 생성 완료: {file_name} (sr={sr})")

if __name__ == "__main__":
    main()
