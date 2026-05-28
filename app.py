import streamlit as st
import os
import json
import time
from datetime import datetime
from urllib.parse import urlparse
from newspaper import Article
import google.genai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(page_title="Verité Claim Spotter", page_icon="🔎", layout="wide")

# Add custom CSS to give it a premium, modern feel that respects Light/Dark mode!
st.markdown("""
<style>
    .claim-card {
        background-color: var(--secondary-background-color);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 20px;
        border-left: 5px solid #2563eb;
    }
    .claim-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-color);
        margin-bottom: 15px;
    }
    .claim-detail {
        margin-bottom: 8px;
        font-size: 0.95rem;
        color: var(--text-color);
    }
    .speaker-badge {
        background: #3b82f6;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 500;
        font-size: 0.85rem;
        display: inline-block;
        margin-bottom: 10px;
    }
    .fact-score {
        float: right;
        background: #f59e0b;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# The EXACT rigorous prompt from claim_spotter_agent.py, with translation added
CLASSIFICATION_PROMPT = '''
You are an expert fact-checker and political analyst working for Verité Research (FactCheck.lk). Your task is to scan news articles and identify statements or claims that are strong candidates for fact-checking.

**CRITICAL INSTRUCTION:** If the provided article text is in Sinhala or Tamil, you MUST automatically translate the extracted claims into English. The final output must be 100% in English.

✅ An article contains a fact-checkable claim if it includes:
1. A direct or indirect quote from a politician, public official, government entity, or influential public figure in Sri Lanka.
2. An assertion of fact that can be objectively verified or debunked using publicly available data (Central Bank, IMF, etc.).
3. The claim must fall into verifiable categories (Economic, Debt, Trade, Labour, Social, Governance, Energy).
4. It must NOT be a purely subjective opinion or vague future promise.

❌ Do NOT flag:
- Statements of intent or policy goals without specific measurable claims
- Pure value judgements or opinions
- Claims that are not verifiable through publicly accessible data

**🎯 Intelligent Output Requirements:**
For each valid claim found in the provided article, provide:
- `speaker`: The person or organization who made the claim (e.g., "Anura Kumara Dissanayake")
- `claim_text`: The exact quote or specific paraphrased claim made by the speaker (in English).
- `context`: Where and when it was said, if mentioned (e.g., "In Parliament").
- `fact_check_potential`: An integer score from 1 to 10 on how verifiable and important this claim is.

**Output Format:** Return ONLY a JSON array of claim objects. If no claims are found, return an empty array `[]`. Do NOT include markdown blocks like ```json.
'''

def get_genai_client():
    api_key = None
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY")
    except FileNotFoundError:
        pass
        
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        
    if not api_key:
        st.error("Missing Google API Key. Please add it to your .env file or Streamlit Secrets.")
        st.stop()
    return genai.Client(api_key=api_key)

def extract_article(url):
    """Use newspaper3k to extract raw article data from the URL, with a BeautifulSoup fallback."""
    text = ""
    title = ""
    publish_date = None
    
    # 1. Try standard newspaper3k extraction with a spoofed User-Agent
    try:
        from newspaper import Config
        config = Config()
        config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        config.request_timeout = 15
        
        article = Article(url, config=config)
        article.download()
        article.parse()
        
        text = article.text
        title = article.title
        publish_date = article.publish_date
    except Exception as e:
        pass # We will let the fallback try!

    # 2. Fallback for Sinhala/Tamil sites if newspaper3k failed or returned < 100 characters
    try:
        if not text or len(text.strip()) < 100:
            import cloudscraper
            from bs4 import BeautifulSoup
            
            # Initialize cloudscraper with realistic browser signatures to bypass firewalls
            scraper = cloudscraper.create_scraper(browser={
                'browser': 'chrome',
                'platform': 'windows',
                'mobile': False
            })
            
            resp = scraper.get(url, timeout=20)
            resp.raise_for_status() # Throw error if it still fails (rare)
            soup = BeautifulSoup(resp.content, 'html.parser')
            
            # Grab all paragraphs
            paragraphs = soup.find_all('p')
            text = '\n'.join([p.get_text(strip=True) for p in paragraphs])
            
            if not title:
                title_tag = soup.find('title')
                title = title_tag.get_text(strip=True) if title_tag else "Unknown Title"
                
        # If both methods failed to get meaningful text
        if not text or len(text.strip()) < 50:
            return {"error": "Website anti-bot protection completely blocked text extraction."}
            
        domain = urlparse(url).netloc.replace('www.', '')
        date_str = str(publish_date.date()) if publish_date else datetime.now().strftime("%Y-%m-%d")
        
        return {
            "title": title or "Unknown Title",
            "text": text,
            "date": date_str,
            "source": domain,
            "url": url
        }
    except Exception as e:
        return {"error": str(e)}

def analyze_claims(text, client):
    """Pass the raw text to Gemini to find and translate claims"""
    if not text or len(text.strip()) < 50:
        return []
        
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{CLASSIFICATION_PROMPT}\n\n**Article Text to Analyze:**\n{text}"
        )
        
        raw_output = response.text.strip()
        if raw_output.startswith("```json"): raw_output = raw_output[7:]
        elif raw_output.startswith("```"): raw_output = raw_output[3:]
        if raw_output.endswith("```"): raw_output = raw_output[:-3]
        
        return json.loads(raw_output)
    except Exception as e:
        st.error(f"AI Analysis Failed: {str(e)}")
        return []

# --- UI Layout ---
st.markdown("""
<div style="text-align: center; padding: 2rem 0; background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%); color: white; border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);">
    <h1 style="margin:0; font-size: 3rem; font-weight: 800; letter-spacing: -1px;">Verité Claim Spotter</h1>
    <p style="font-size: 1.2rem; opacity: 0.9; margin-top: 10px;">Instantly extract fact-checkable claims from any news article in English, Sinhala, or Tamil.</p>
</div>
""", unsafe_allow_html=True)

# Input Section
st.markdown("### 🔗 Analyze a New Article")
st.markdown("<p style='color: var(--text-color); opacity: 0.8; margin-bottom: 1rem; font-size: 0.95rem;'>Paste the direct URL of any Sri Lankan news article below. The AI will automatically bypass the ads, translate the text to English, and extract fact-checkable claims.</p>", unsafe_allow_html=True)

# Callback to clear the input
def clear_input():
    st.session_state.url_input_box = ""

url_input = st.text_input("Paste URL", placeholder="https://www.dailymirror.lk/...", label_visibility="collapsed", key="url_input_box")

col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
with col2:
    analyze_btn = st.button("Extract Claims ✨", type="primary", use_container_width=True)
with col3:
    clear_btn = st.button("Clear 🗑️", use_container_width=True, on_click=clear_input)

if analyze_btn and url_input:
    client = get_genai_client()
    
    # State variables to hold results outside the status context
    article_data = None
    claims = None
    start_time = time.time()
    
    # Use the beautiful st.status widget for loading
    with st.status("Analyzing Article...", expanded=True) as status:
        st.write("🌐 Fetching webpage contents...")
        article_data = extract_article(url_input)
        
        if "error" in article_data:
            status.update(label="Analysis Failed", state="error", expanded=True)
            st.error(f"Failed to read the article URL: {article_data['error']}")
        else:
            st.write("🧠 Passing to Gemini AI for translation and extraction...")
            claims = analyze_claims(article_data['text'], client)
            status.update(label=f"Analysis Complete! ({time.time() - start_time:.1f}s)", state="complete", expanded=False)
            
    # Render Results OUTSIDE the status accordion so they are always visible
    if article_data and "error" not in article_data:
        # Display Metadata
        st.markdown(f"**📰 Source:** {article_data['source']} &nbsp;&nbsp;|&nbsp;&nbsp; **📅 Date:** {article_data['date']}")
        st.markdown(f"**📑 Title:** {article_data['title']}")
        st.divider()
        
        # Display Claims
        if not claims:
            st.info("No verifiable political or economic claims found in this article.")
        else:
            st.success(f"Found {len(claims)} fact-checkable claims!")
            
            for claim in claims:
                st.markdown(f"""
                <div class="claim-card">
                    <div class="speaker-badge">🗣 {claim.get('speaker', 'Unknown Speaker')}</div>
                    <div class="fact-score">Score: {claim.get('fact_check_potential', 0)}/10</div>
                    <div class="claim-title">"{claim.get('claim_text', '')}"</div>
                    <div class="claim-detail"><strong>📌 Context:</strong> {claim.get('context', 'N/A')}</div>
                </div>
                """, unsafe_allow_html=True)
