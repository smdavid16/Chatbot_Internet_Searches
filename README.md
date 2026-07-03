# Chatbot Specializat in Cautari pe Internet

## Funcționalități Principale

- **Căutare pe Internet (SearXNG)**: Folosește o instanță locală de SearXNG (rulată prin Docker) pentru a extrage și sumariza rezultatele web. Această abordare evită erorile 403, limitările de trafic și testele CAPTCHA.
- **Vremea în Timp Real**: Integrează API-ul **OpenWeatherMap** pentru a oferi date precise despre starea vremii din orice oraș sau țară.
- **Cache Semantic (ChromaDB)**: Memorează rezultatele căutărilor anterioare într-o bază de date vectorială locală. Folosește căutarea semantică pentru a returna rapid răspunsuri la întrebări similare, economisind timp și resurse.
- **Sincronizare cu Timpul Real**: Folosește un server NTP pentru a răspunde cu exactitate la întrebări legate de data și ora curente din orice fus orar.
- **Mecanism de Siguranță ("Nudge")**: Previne "halucinațiile" modelului forțându-l să folosească instrumentele de căutare atunci când detectează întrebări despre date factuale (persoane, scoruri, prețuri etc.).

## Cerințe Preliminare

- [Python 3.10+](https://www.python.org/)
- [Ollama](https://ollama.com/) (cu modelul `qwen2.5:3b` descărcat: rulează `ollama pull qwen2.5:3b`)
- [Docker & Docker Compose](https://www.docker.com/) (pentru a rula motorul privat de căutare SearXNG)

## Instalare

1. **Clonează repozitoriul și accesează directorul:**
   ```bash
   cd Chatbot
   ```

2. **Creează și activează un mediu virtual:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instalează dependențele:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Pornește instanța locală SearXNG:**
   ```bash
   docker compose up -d
   ```

5. **Configurează variabilele de mediu:**
   - Deschide sau redenumește fișierul `.env.example` în `.env`.
   - Adaugă cheia ta pentru API-ul de vreme:
     ```env
     OPENWEATHERMAP_API_KEY=cheia_ta_api_aici
     ```

## Utilizare

Pentru a iniția conversația cu asistentul, rulează:

```bash
python chatbot.py
```

### Comenzi Speciale (în timpul conversației)
- `/clear-cache`: Șterge toate rezultatele salvate în memoria cache a bazei de date.
- `/cache-stats`: Afișează statistici despre intrările din cache.
- `quit` sau `exit`: Închide programul.

## Structura Proiectului

- `chatbot.py`: Punctul de intrare care conține agentul principal și funcțiile instrumentelor (tools).
- `search_cache.py`: Logica de memorare și căutare semantică folosind ChromaDB.
- `config.py`: Parametrii și configurările globale.
- `docker-compose.yml` & `searxng_settings.yml`: Configurația Docker pentru a asigura funcționarea independentă a motorului de căutare.
