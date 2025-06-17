import requests
import os

# 서버 주소 설정
HOST = "127.0.0.1"
PORT = 9880
BASE_URL = f"http://{HOST}:{PORT}"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def send_control_command(command: str):
    """
    control 명령 보내기: 'restart' 또는 'exit'
    """
    url = f"{BASE_URL}/control"
    params = {"command": command}
    response = requests.get(url, params=params)

    if response.status_code == 200 or response.status_code == 204:
        print(f"서버 명령어 '{command}' 전송 성공")
    else:
        print(f"서버 명령어 실패: {response.status_code}")
        print(response.text)

def set_gpt_weights(weights_path: str):
    """
    GPT 가중치 변경
    """
    url = f"{BASE_URL}/set_gpt_weights"
    params = {"weights_path": weights_path}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        print("✅ GPT 가중치 설정 성공")
    else:
        print(f"❌ GPT 가중치 설정 실패: {response.status_code}")
        print(response.text)

def set_sovits_weights(weights_path: str):
    """
    So-VITS 가중치 변경
    """
    url = f"{BASE_URL}/set_sovits_weights"
    params = {"weights_path": weights_path}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        print("✅ SoVITS 가중치 설정 성공")
    else:
        print(f"❌ SoVITS 가중치 설정 실패: {response.status_code}")
        print(response.text)

def request_tts(
    text: str,
    text_lang: str,
    ref_audio_path: str,
    prompt_text: str,
    prompt_lang: str,
    media_type: str = "wav",
    streaming_mode: bool = False,
    save_path: str = "output.wav"
):
    """
    /tts API로 텍스트 음성 합성 요청 후 파일 저장
    """
    url = f"http://127.0.0.1:9880/tts"

    payload = {
        "text": text,
        "text_lang": text_lang,
        "ref_audio_path": ref_audio_path,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang,
        "media_type": media_type,
        "streaming_mode": streaming_mode
    }

    try:
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            print(f"✅ 음성 저장 완료: {save_path}")
        else:
            print(f"❌ 요청 실패: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"⚠️ 예외 발생: {e}")

gpt_ckpt_path = os.path.join(BASE_DIR, "models", "Hutao_GPT.ckpt")
sovits_pth_path = os.path.join(BASE_DIR, "models", "Hutao_Sovits.pth")

set_gpt_weights(gpt_ckpt_path)
set_sovits_weights(sovits_pth_path)

request_tts(
    text="안녕, 만나서 반가워!",
    text_lang="ko",
    ref_audio_path=f"{BASE_DIR}\\example.wav",
    prompt_text="그나저나, 클로린드가 옛날에 푸리나의 부하였나봐?",
    prompt_lang="ko",
    media_type="wav",
    save_path="output.wav"
)