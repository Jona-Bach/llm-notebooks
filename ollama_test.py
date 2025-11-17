import requests
import json
import time

OLLAMA_URL = "http://localhost:11434"


def pull_model(model: str):
    """
    Lädt ein Modell über den Ollama-Pull-Endpoint.
    Entspricht: `ollama pull <model>`.
    """
    url = f"{OLLAMA_URL}/api/pull"
    payload = {"name": model}

    print(f"Pulling model '{model}' von {url} ...")

    resp = requests.post(url, json=payload, stream=True)
    print("HTTP-Status:", resp.status_code)

    if resp.status_code != 200:
        print("Antwort-Body:")
        print(resp.text[:1000])
        resp.raise_for_status()

    for line in resp.iter_lines():
        if line:
            data = json.loads(line.decode("utf-8"))
            print(data)

    print("✅ Model pull completed")


def generate(model: str, prompt: str):
    """
    Versucht zuerst den nativen Ollama-Endpoint /api/generate (Streaming).
    Falls der 404 liefert, wird automatisch auf den OpenAI-kompatiblen
    Endpoint /v1/chat/completions gewechselt.
    
    Gibt (antwort_text, antwortzeit_in_sekunden) zurück.
    """

    native_url = f"{OLLAMA_URL}/api/generate"
    native_payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
    }

    print(f"POST {native_url}")
    start = time.perf_counter()
    resp = requests.post(native_url, json=native_payload, stream=True)
    first_elapsed = time.perf_counter() - start

    if resp.status_code == 200:
        full_response = ""
        for line in resp.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                full_response += data.get("response", "")
        return full_response, first_elapsed

    if resp.status_code != 404:
        print("❌ Fehler von /api/generate:")
        print(resp.status_code, resp.text[:500])
        resp.raise_for_status()

    print("ℹ️ /api/generate nicht gefunden (404). Fallback auf /v1/chat/completions ...")


    oa_url = f"{OLLAMA_URL}/v1/chat/completions"
    oa_payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
    }

    print(f"POST {oa_url}")
    start = time.perf_counter()
    oa_resp = requests.post(oa_url, json=oa_payload)
    elapsed = time.perf_counter() - start

    if oa_resp.status_code != 200:
        print("❌ Fehler von /v1/chat/completions:")
        print(oa_resp.status_code, oa_resp.text[:500])
        oa_resp.raise_for_status()

    data = oa_resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        content = str(data)

    return content, elapsed


if __name__ == "__main__":
    tags = requests.get(f"{OLLAMA_URL}/api/tags")
    print("Tags-Status:", tags.status_code)
    print("Antwort-Body:", tags.text[:200], "...\n")

    
    model_name = "llama3.2:3b" 



    antwort, dauer = generate(model_name, "Erklär mir in 2 Sätzen, was ein LLM ist.")
    print("\nAntwort:")
    print(antwort)
    print(f"\n Antwortzeit: {dauer:.3f} Sekunden")



    # Hier Unten einkommentieren für Modell download
    #pull_model("llama3.1:8b")
    #pull_model("gpt-oss:latest")