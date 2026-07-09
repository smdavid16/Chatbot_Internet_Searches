# 🌐 Chatbot cu Căutare pe Internet

Un chatbot CLI propulsat de **Ollama (Qwen2.5 3B)** cu posibilitatea de a căuta pe internet în timp real, cu indexare locală a știrilor românești, vreme în timp real și traducere automată Română ↔ Engleză — totul rulând 100% local.

Modelul decide *când* să caute — invocând utilitare doar atunci când are nevoie de informații la zi sau de date factuale pe care nu le cunoaște deja.

---

## ✨ Funcționalități

| Funcționalitate | Descriere |
|---|---|
| **Căutare Google (SearXNG)** | Căutare privată pe web, auto-găzduită prin intermediul unui container local de Docker cu SearXNG. Fără chei API, fără CAPTCHA, fără limite de solicitări. |
| **Bază de Date Locală de Știri** | Extrage, traduce și indexează articole de pe [Biziday.ro](https://www.biziday.ro/) într-un vector store ChromaDB pentru regăsire semantică instantanee. |
| **Cache Semantic pentru Căutări** | Salvează rezultatele căutărilor în ChromaDB. Interogările ulterioare similare sunt returnate instantaneu fără a mai accesa internetul. |
| **Vreme în Timp Real** | Preia date meteo la zi de la **OpenWeatherMap** pentru orice oraș din lume. |
| **Sincronizare Timp NTP** | Interoghează un server NTP pentru a obține data/ora precisă în orice fus orar IANA. |
| **Traducere Română ↔ Engleză** | Detecție automată a limbii și traducere utilizând modelul **SeamlessM4T** de la Meta (rulează local pe GPU/CPU). |
| **Prevenirea Halucinațiilor** | Detectează întrebările factuale și forțează modelul să folosească utilitarele de căutare în loc să răspundă din datele de antrenament (potențial învechite). |
| **Apelare de Unelte (Tool Calling)** | LLM-ul decide autonom ce utilitare să invoce prin intermediul interfeței native de tool-calling din Ollama. |

---

## 🏗️ Arhitectură

```
┌─────────────────────────────────────────────────────────────────┐
│                         Utilizator (CLI)                        │
│                Input în Română sau Engleză                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │ translator  │  SeamlessM4T (Română ↔ Engleză)
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  chatbot.py │  Buclă agent + orchestrare utilitare
                    │  (Ollama)   │  Qwen2.5:3b cu funcție de tool calling
                    └──┬───┬───┬──┘
                       │   │   │
          ┌────────────┘   │   └────────────────┐
          │                │                    │
   ┌──────▼──────┐  ┌─────▼──────┐   ┌─────────▼─────────┐
   │ SearXNG     │  │ Biziday    │   │ Vreme / Ora       │
   │ (Docker)    │  │ Index      │   │ (APIs / NTP)      │
   │             │  │ ChromaDB   │   │                   │
   └──────┬──────┘  └─────┬──────┘   └───────────────────┘
          │               │
   ┌──────▼──────┐  ┌─────▼──────┐
   │ Cache de    │  │ scrape_    │
   │ Căutare     │  │ biziday.py │
   │ (ChromaDB)  │  └────────────┘
   └─────────────┘
```

### Fluxul Datelor

1. **Input utilizator** → Detecția limbii → Datele de intrare în limba română sunt traduse în engleză.
2. **Preluare anticipată (Pre-fetch)** → Baza de date locală de știri Biziday este interogată automat pentru articole relevante.
3. **Raționament LLM** → Modelul Qwen2.5:3b de la Ollama procesează interogarea cu definițiile uneltelor. Poate apela:
   - `google_search` — căutare web via SearXNG
   - `scrape_webpage` — citire în profunzime a unui URL specific
   - `search_biziday_news` — căutare semantică prin articolele de știri indexate
   - `get_current_weather` — API-ul OpenWeatherMap
   - `get_current_datetime` — serverul de timp NTP
4. **Execuția uneltei** → Rezultatele sunt trimise înapoi modelului pentru sinteză.
5. **Răspuns** → Dacă utilizatorul a vorbit în română, răspunsul este tradus înapoi în limba română.

---

## 📋 Cerințe Preliminare

- **Python 3.10+**
- **[Ollama](https://ollama.com/)** — cu modelul `qwen2.5:3b` descărcat:
  ```bash
  ollama pull qwen2.5:3b
  ```
- **[Docker & Docker Compose](https://www.docker.com/)** — pentru containerul motorului de căutare SearXNG
- **Placă video NVIDIA (GPU)** *(recomandat)* — Traducerea cu SeamlessM4T beneficiază de accelerare GPU; trece automat pe CPU dacă nu este disponibilă

---

## 🚀 Instalare

### 1. Clonare și accesare repository

```bash
git clone <repository-url>
cd Chatbot
```

### 2. Crearea și activarea unui mediu virtual

```bash
python3 -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows
```

### 3. Instalarea dependențelor Python

```bash
pip install -r requirements.txt
```

**Dependențe:**

| Pachet | Scop |
|---|---|
| `ollama` | Client Python pentru platforma LLM Ollama |
| `requests` | Cereri HTTP pentru extragere date web (scraping) și API-uri |
| `beautifulsoup4` | Parsare HTML și extragere de conținut |
| `ntplib` | Interogări servere de timp NTP |
| `chromadb` | Bază de date vectorială pentru cache-ul semantic și indexul de știri |
| `sentence-transformers` | Model de embedding folosit de ChromaDB (all-MiniLM-L6-v2) |
| `python-dotenv` | Încărcare variabile de mediu din fișierul `.env` |
| `transformers` | Hugging Face Transformers pentru SeamlessM4T |
| `torch` | Framework de deep learning PyTorch |
| `sentencepiece` | Tokenizer necesar pentru modelul SeamlessM4T |

### 4. Pornirea motorului local de căutare SearXNG

```bash
docker compose up -d
```

Acest pas lansează o instanță privată de SearXNG la `http://localhost:8080`.

### 5. Configurarea variabilelor de mediu

Copiați fișierul exemplu și completați-l cu cheile dumneavoastră:

```bash
cp .env.example .env
```

Editați `.env`:

```env
OPENWEATHERMAP_API_KEY=cheia_ta_api_aici    # Necesar pentru utilitarul meteo
HF_TOKEN=tokenul_tau_huggingface_aici       # Necesar pentru descărcarea modelului SeamlessM4T
```

| Variabilă | Necesar | Descriere |
|---|---|---|
| `OPENWEATHERMAP_API_KEY` | Opțional | Activează utilitarul meteo. Obțineți o cheie gratuită la [openweathermap.org](https://openweathermap.org/api). |
| `HF_TOKEN` | Opțional | Token Hugging Face pentru a descărca modelul de traducere SeamlessM4T la prima rulare. |

---

## 💬 Utilizare

### Pornirea chatbot-ului

```bash
python chatbot.py
```

La prima rulare, chatbot-ul va:
1. Încărca modelul de traducere SeamlessM4T (descarcă ~2-5 GB la prima rulare).
2. Sincroniza cele mai recente 20 de articole de pe Biziday.ro în baza de date locală.
3. Afișa un mesaj de întâmpinare și va aștepta inputul dumneavoastră.

### Comenzi de chat

| Comandă | Descriere |
|---|---|
| `/clear-cache` | Curăță toate rezultatele căutărilor salvate în ChromaDB |
| `/cache-stats` | Afișează statistici despre cache (număr intrări, TTL, prag de similaritate) |
| `quit` / `exit` / `q` | Închide chatbot-ul |

---

## 🛠️ Utilitare Independente

Mai multe module pot fi rulate independent din linia de comandă.

### Extragere articole Biziday

```bash
# Extrage 20 de articole de pe prima pagină și le salvează în JSON
python scrape_biziday.py

# Extrage 10 articole și le salvează într-o locație personalizată
python scrape_biziday.py --count 10 --output /cale/catre/iesire.json

# Extrage și indexează imediat în ChromaDB
python scrape_biziday.py --index
```

### Indexare articole în ChromaDB

```bash
# Indexează articole din fișierul JSON implicit
python index_biziday.py

# Indexează dintr-un fișier specific
python index_biziday.py --input cale/catre/fisier.json

# Caută prin articolele indexate
python index_biziday.py --search "summit NATO"

# Arată statisticile colecției
python index_biziday.py --stats

# Șterge toate articolele indexate
python index_biziday.py --clear
```

### Curățare HTML din articole

```bash
# Reface și curăță textul articolelor din fișierul JSON implicit
python clean_biziday.py

# Curăță un anumit fișier JSON
python clean_biziday.py cale/catre/stiriBiziday.json
```

---

## 📁 Structura Proiectului

```
Chatbot/
├── chatbot.py               # Punctul principal de intrare — buclă agent, funcții unelte, CLI
├── config.py                # Toate constantele de configurare și variabilele de mediu
├── search_cache.py          # Cache-ul semantic pentru căutare (susținut de ChromaDB)
├── translator.py            # Traducere Română ↔ Engleză (SeamlessM4T)
├── index_biziday.py         # Indexator articole Biziday + căutare în ChromaDB
├── scrape_biziday.py        # Extragere date de pe pagina principală Biziday.ro
├── clean_biziday.py         # Post-procesor pentru a curăța codul HTML brut al articolelor
├── requirements.txt         # Dependențe Python
├── .env.example             # Șablon pentru variabilele de mediu
├── docker-compose.yml       # Configurație container Docker SearXNG
├── searxng_settings.yml     # Setări pentru motorul SearXNG
├── settings.yml             # Configurații adiționale pentru SearXNG
└── stiriBiziday.json        # Articole extrase stocate în cache (generate automat)
```

### Detalii despre module

#### `chatbot.py` — Chatbot-ul Principal & Bucla Agentului

Nucleul aplicației. Conține:

- **Funcțiile uneltelor** expuse către LLM via tool-calling-ul din Ollama:
  - `google_search(query)` — Caută pe web prin intermediul instanței locale SearXNG, descarcă fiecare pagină cu rezultate, extrage fragmente relevante bazat pe o scorare de cuvinte cheie și returnează rezultate structurate. Rezultatele sunt stocate în ChromaDB.
  - `scrape_webpage(url, query)` — Descarcă un URL specific și extrage conținutul relevant. Util pentru a citi în detaliu o pagină găsită în rezultatele căutării.
  - `get_current_datetime(timezone_name)` — Interoghează un server NTP (`pool.ntp.org`) pentru ora exactă în orice fus orar IANA. Include potrivire aproximativă a fusului orar (ex. "tokyo" se asociază cu "Asia/Tokyo").
  - `get_current_weather(location)` — Preluare date meteo de la API-ul OpenWeatherMap pentru orice oraș.
  - `search_biziday_news(query)` — Caută semantic prin articolele de știri românești indexate local.

- **Bucla Agentului** (`agent_turn`) — Apelează repetitiv LLM-ul, execută utilitarele cerute, reintroduce rezultatele înapoi și repetă până când modelul produce răspunsul text final (până la limita de `MAX_TOOL_ITERATIONS` runde).

- **Prevenirea Halucinațiilor** — Dacă modelul încearcă să răspundă la o întrebare factuală fără a utiliza nicio unealtă, sistemul detectează acest comportament prin potrivire de cuvinte cheie și obligă modelul să folosească mai întâi un utilitar.

- **Pre-fetch Biziday** — Înaintea oricărui apel către LLM, sistemul caută automat în baza de date locală de știri. Dacă sunt găsite articole relevante (în limita `BIZIDAY_RELEVANCE_THRESHOLD`), acestea sunt injectate în conversație ca mesaj de sistem.

- **Funcții ajutătoare interne**:
  - `_searxng_search()` — Interoghează API-ul JSON SearXNG.
  - `_fetch_page()` — Descarcă și parsează o pagină web cu o extragere inteligentă a conținutului (pune pe primul plan elementele `<article>`, `<main>`, structurile comune (div) de conținut, iar ca rezervă o euristică pentru a identifica cel mai mare div).
  - `_extract_relevant_snippets()` — Acordă un scor propozițiilor în funcție de prezența cuvintelor cheie din interogare și le returnează pe cele mai relevante sub formă de listă.

#### `search_cache.py` — Cache Semantic pentru Căutări

O clasă `SearchCache` care utilizează ChromaDB pentru a stoca perechile interogare-rezultat:

- **Potrivire semantică** — Folosește modelul pre-integrat în ChromaDB `all-MiniLM-L6-v2` pentru a face potriviri pe bază de similaritate. O căutare după "președintele româniei" va duce la un hit pe rezultatul memorat pentru "cine conduce românia", dacă ambele se încadrează în distanța L2 admisă.
- **Expirare TTL** — Intrările stocate expiră automat după o perioadă configurabilă (implicit: 24 de ore).
- **API**: `lookup(query)`, `store(query, result)`, `clear()`, `stats()`

#### `translator.py` — Traducere Română ↔ Engleză

Asigură traducere automată utilizând modelul SeamlessM4T de la Meta:

- **Încărcarea modelului** — Încearcă mai întâi modelul "Large v2" (2.3B parametri). Dacă memoria GPU este insuficientă, trece la modelul "Medium" (1.2B parametri). Se încarcă doar la prima nevoie.
- **Detecția limbii** — Se bazează pe o euristică rapidă utilizând diacritice românești (ă, â, î, ș, ț) și o listă specifică de cuvinte marcator (marker words). Se activează dacă găsește diacritice sau dacă 15%+ din text este format din marker words românești.
- **Traducere segmentată** — Textele lungi sunt tăiate în porțiuni (maxim 800 de caractere), rupte în funcție de granițele paragrafelor, ale liniilor, apoi ale propozițiilor. Fiecare secvență e tradusă independent, apoi textul este reasamblat.
- **Înlăturarea Markdown-ului** — Înainte de traducerea din engleză în română a răspunsului dat de model, formatarea markdown este înlăturată (bold, italic, links, headings) pentru a nu periclita ieșirea sistemului de traducere.

#### `index_biziday.py` — Indexatorul de Articole de Știri

Gestionează colecția ChromaDB cu articole de știri Biziday traduse:

- **Indexare în masă** (`index_articles`) — Citește un fișier JSON de articole românești, traduce fiecare titlu și conținut în limba engleză și le introduce în ChromaDB alături de metadate complete.
- **Sincronizare live** (`sync_latest_articles`) — Extrage datele de pe prima pagină Biziday, verifică dacă URL-urile articolelor se găsesc deja în baza de date cu ajutorul unui hash și le indexează exclusiv pe cele noi. Adaugă, de asemenea, articolele noi la fișierul JSON de backup.
- **Căutare semantică** (`search_biziday`) — Interoghează colecția ChromaDB folosind similaritatea de embedding-uri și returnează rezultate ce conțin titluri, link-uri, date de publicare, conținutul și scorul de distanță.
- **Managementul Colecției** — `clear_collection()` și `collection_stats()` pentru întreținere.
- **Eliminare duplicate** — Folosește o semnătură hash (SHA-256) a URL-ului articolului drept ID de document pentru a preveni duplicarea.

#### `scrape_biziday.py` — Extragere de pe Pagina Principală (Scraper)

Preia paginile index de pe Biziday.ro pentru a trage link-urile și titlurile articolelor:

- Analizează structura `<ul class="loop">` pentru a localiza itemi individuali de articole, eludând textele publicitare (ads).
- `extract_article_text(url)` citește paginile independente și extrage materialul scris din etichetele `<p>`, `<h1>`, `<h2>`, `<h3>` stocate în div-ul central cu conținut.

#### `clean_biziday.py` — Curățător HTML pentru Articole

Un script auxiliar care accesează și curăță secțiunea `HTML_Sursa` din fișierul JSON compilat anterior de către scraper, transformând HTML-ul brut în text extras curat.

#### `config.py` — Configurație

Toate datele și parametrii modificabili se află aici. Acesta preia și cheile API din `.env` grație bibliotecii `python-dotenv`.

---

## ⚙️ Referința Configurărilor

Toate setările sunt definite în `config.py`:

| Setare | Valoare Implicită | Descriere |
|---|---|---|
| `MODEL_NAME` | `qwen2.5:3b` | Modelul Ollama folosit pentru discuții |
| `NUM_SEARCH_RESULTS` | `3` | Numărul de rezultate per interogare pe web |
| `SEARXNG_URL` | `http://localhost:8080` | URL-ul instanței locale de SearXNG |
| `MAX_PAGE_CONTENT_LENGTH` | `5000` | Max caractere extrase dintr-o singură pagină web |
| `PAGE_REQUEST_TIMEOUT` | `10` | Limita de timp pentru o cerere HTTP (secunde) |
| `MAX_TOOL_ITERATIONS` | `10` | Limită de siguranță la apelurile de utilitare per tură de agent |
| `CACHE_DIR` | `~/.chatbot_cache/chroma_db` | Locație de memorie persistentă pentru ChromaDB |
| `CACHE_TTL_SECONDS` | `86400` (24h) | Cât timp este valid un rezultat de căutare memorat în cache |
| `CACHE_SIMILARITY_THRESHOLD` | `0.35` | Distanța L2 maximă pentru a obține o potrivire în cache (mai mic = mai strict) |
| `BIZIDAY_COLLECTION_NAME` | `biziday_news` | Numele colecției din ChromaDB pentru știrile Biziday |
| `BIZIDAY_SEARCH_RESULTS` | `5` | Numărul implicit pentru rezultatele de căutare pentru știri |
| `BIZIDAY_RELEVANCE_THRESHOLD` | `1.0` | Distanța L2 maximă pentru a considera un fragment de știre relevant |
| `SEAMLESS_MODEL_LARGE` | `facebook/seamless-m4t-v2-large` | Model primar de traducere (2.3B parametri) |
| `SEAMLESS_MODEL_FALLBACK` | `facebook/seamless-m4t-medium` | Model secundar (rezervă) de traducere (1.2B parametri) |

---

## 🧠 Cum Funcționează — Privește în Detaliu

### Mecanismul de Apelare de Utilitare (Tool-Calling)

Chatbot-ul folosește funcționalitatea nativă din Ollama pentru **tool-calling**. Funcțiile Python însoțite de type hints și de șiruri de documentare (docstrings) trec direct prin API-ul `ollama.chat()` în parametrul `tools`. Ollama citește semnăturile acestor funcții și generează automat definiții în format JSON schema. Apoi, LLM-ul va întoarce o variabilă structurată `tool_calls` în răspunsul său atunci când decide să folosească un utilitar.

### Ciclul de Căutare

1. Utilizatorul adresează o întrebare (e.g. „Care este PIB-ul României?”).
2. Interogarea este verificată în **cache-ul semantic** (ChromaDB) pentru interogări anterioare similare.
3. Dacă nu există în cache, se apelează SearXNG pentru rezultate web.
4. Fiecare URL rezultat este preluat, și HTML-ul descărcat este divizat prin BeautifulSoup:
   - Elementele fără conținut util (bara de navigare, subsol, scripturi, header) sunt eliminate.
   - Conținutul este extras din `<article>` > `<main>` > containere specifice de date > cel mai mare div > body.
   - Metadatele (titlu, meta description, titluri) sunt inserate la început.
5. Propozițiile din textul extras sunt **punctate pe baza potrivirii cuvintelor cheie** din interogare.
6. Propozițiile cu scorurile cele mai mari sunt returnate ca elemente punctate de tip „KEY FACTS”.
7. Rezultatele sunt stocate în cache-ul semantic pentru căutări viitoare.

### Procesul de Traducere

1. Textul de intrare suferă o verificare pentru a se identifica prezența de **diacritice românești** (ă, â, î, ș, ț) — un semnal instantaneu.
2. În lipsa diacriticelor, se numără **cuvintele marcatoare ale limbii române**. Dacă cel puțin 15% dintre cuvinte se potrivesc, este clasificat ca fiind în română.
3. Pentru Română → Engleză: textul este tradus direct într-o singură secvență.
4. Pentru Engleză → Română (traducerea răspunsului final):
   - Tiparele Markdown sunt înlăturate complet (bold, italic, link-uri, titluri).
   - Textul este divizat în porțiuni de maxim 800 de caractere la granițele paragrafelor, ale liniilor și ale propozițiilor.
   - Fiecare segment este tradus independent de SeamlessM4T.
   - Fragmentele traduse sunt reasamblate cu întreruperi de paragraf.

### Procesul Bazei de Date cu Știri

1. La lansarea chatbot-ului, `sync_latest_articles()` extrage cele mai noi articole de pe prima pagină Biziday.ro.
2. Fiecărui URL al articolului i se atribuie un identificator hash (SHA-256) și este verificat în ChromaDB.
3. Articolele noi sunt procesate pentru a prelua textul complet, acesta fiind tradus în engleză și apoi actualizat sau introdus.
4. Fișierul `.json` de backup este actualizat simultan pentru a păstra permanent o copie la nivel local.
5. Pe parcursul unei discuții chat, orice mesaj venit de la utilizator provoacă un **pre-fetch** ce interoghează baza de știri românești, trimițând orice articol apropiat sub formă de notițe suplimentare de context înaintea rulării modelului.

---

## 📄 Licență

Acest proiect se distribuie ca atare (as-is), exclusiv cu scop educațional și orientat către uz personal.
