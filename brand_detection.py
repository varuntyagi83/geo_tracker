"""
Brand Detection Module for GEO Tracker.

LLM-BASED APPROACH:
- Uses a fast LLM (GPT-4o-mini or Gemini Flash) to extract actual brand names
- The LLM understands context and knows what are real brands vs generic words
- No more false positives like "India", "Berlin", "The", etc.
"""

import os
import json
import re
from typing import Set, List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout


# ============================================
# LLM BRAND EXTRACTION
# ============================================

def _call_openai_for_brands(text: str, industry: str, market: str, our_brand: str) -> Set[str]:
    """Use OpenAI GPT-4o-mini to extract brand names."""
    try:
        from openai import OpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return set()
        
        client = OpenAI(api_key=api_key)
        
        prompt = f"""Extract ONLY actual company/brand names from the following text.

CONTEXT:
- Industry: {industry}
- Market/Country: {market}
- Our brand (exclude this): {our_brand}

RULES:
1. Return ONLY real company names and brand names
2. DO NOT include:
   - Country names (India, Germany, USA, etc.)
   - City names (Berlin, Mumbai, Delhi, etc.)
   - Generic words (Food, Delivery, Restaurant, Quality, etc.)
   - Adjectives (Best, Top, Popular, Indian, German, etc.)
   - Our brand: {our_brand}
3. Include competitor brands, restaurant chains, delivery apps, etc.
4. Return as JSON array of strings, nothing else

TEXT TO ANALYZE:
{text[:3000]}

Return ONLY a JSON array like: ["Brand1", "Brand2", "Brand3"]
If no brands found, return: []"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500,
            timeout=10
        )
        
        result = response.choices[0].message.content.strip()
        
        # Parse JSON response
        # Remove markdown code blocks if present
        if result.startswith("```"):
            result = re.sub(r'^```json?\s*', '', result)
            result = re.sub(r'\s*```$', '', result)
        
        brands = json.loads(result)
        
        if isinstance(brands, list):
            # Filter out our brand and clean up
            our_brand_lower = our_brand.lower()
            our_brand_words = set(our_brand.lower().split())
            
            cleaned = set()
            for b in brands:
                if isinstance(b, str) and b.strip():
                    b = b.strip()
                    if b.lower() != our_brand_lower and b.lower() not in our_brand_words:
                        cleaned.add(b)
            return cleaned
        
        return set()
        
    except Exception as e:
        print(f"[brand_detection] OpenAI extraction error: {e}")
        return set()


def _call_gemini_for_brands(text: str, industry: str, market: str, our_brand: str) -> Set[str]:
    """Use Gemini Flash to extract brand names."""
    try:
        import google.generativeai as genai
        
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return set()
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        prompt = f"""Extract ONLY actual company/brand names from the following text.

CONTEXT:
- Industry: {industry}
- Market/Country: {market}
- Our brand (exclude this): {our_brand}

RULES:
1. Return ONLY real company names and brand names
2. DO NOT include:
   - Country names (India, Germany, USA, etc.)
   - City names (Berlin, Mumbai, Delhi, etc.)
   - Generic words (Food, Delivery, Restaurant, Quality, etc.)
   - Adjectives (Best, Top, Popular, Indian, German, etc.)
   - Our brand: {our_brand}
3. Include competitor brands, restaurant chains, delivery apps, etc.
4. Return as JSON array of strings, nothing else

TEXT TO ANALYZE:
{text[:3000]}

Return ONLY a JSON array like: ["Brand1", "Brand2", "Brand3"]
If no brands found, return: []"""

        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0,
                max_output_tokens=500,
            )
        )
        
        result = response.text.strip()
        
        # Remove markdown code blocks if present
        if result.startswith("```"):
            result = re.sub(r'^```json?\s*', '', result)
            result = re.sub(r'\s*```$', '', result)
        
        brands = json.loads(result)
        
        if isinstance(brands, list):
            our_brand_lower = our_brand.lower()
            our_brand_words = set(our_brand.lower().split())
            
            cleaned = set()
            for b in brands:
                if isinstance(b, str) and b.strip():
                    b = b.strip()
                    if b.lower() != our_brand_lower and b.lower() not in our_brand_words:
                        cleaned.add(b)
            return cleaned
        
        return set()
        
    except Exception as e:
        print(f"[brand_detection] Gemini extraction error: {e}")
        return set()


def extract_brands_with_llm(
    text: str,
    industry: str = "",
    market: str = "",
    our_brand: str = "",
    provider: str = "auto"
) -> Set[str]:
    """
    Extract brand names from text using an LLM.
    
    Args:
        text: The text to analyze
        industry: The industry context (e.g., "Food Delivery", "Supplements")
        market: The market/country (e.g., "India", "Germany")
        our_brand: Our brand name to exclude from results
        provider: "openai", "gemini", or "auto" (tries both)
    
    Returns:
        Set of detected brand names
    """
    if not text or len(text.strip()) < 20:
        return set()
    
    # Try the specified provider or auto-detect
    if provider == "openai":
        return _call_openai_for_brands(text, industry, market, our_brand)
    elif provider == "gemini":
        return _call_gemini_for_brands(text, industry, market, our_brand)
    else:
        # Auto: Try OpenAI first, fallback to Gemini
        brands = _call_openai_for_brands(text, industry, market, our_brand)
        if not brands:
            brands = _call_gemini_for_brands(text, industry, market, our_brand)
        return brands


# ============================================
# MAIN API FUNCTIONS (for compatibility)
# ============================================

# Store context for brand detection (set by the service layer)
_extraction_context = {
    "industry": "",
    "market": "",
}


def set_extraction_context(industry: str = "", market: str = ""):
    """Set the context for brand extraction (called by service layer)."""
    global _extraction_context
    _extraction_context["industry"] = industry
    _extraction_context["market"] = market


def detect_competitor_brands(
    response_text: str,
    sources: List[Dict],
    our_brand: str,
    industry: str = "",
    market: str = "",
) -> Set[str]:
    """
    Main function to detect competitor brands from LLM response.
    
    Uses an LLM to intelligently extract actual brand names,
    avoiding false positives like country names, city names, etc.
    
    Args:
        response_text: The LLM response text
        sources: List of source dictionaries (not used in LLM approach)
        our_brand: Our brand name to exclude
        industry: Industry context (optional, uses stored context if not provided)
        market: Market context (optional, uses stored context if not provided)
    
    Returns:
        Set of competitor brand names
    """
    # Use provided context or fall back to stored context
    ind = industry or _extraction_context.get("industry", "")
    mkt = market or _extraction_context.get("market", "")
    
    return extract_brands_with_llm(
        text=response_text,
        industry=ind,
        market=mkt,
        our_brand=our_brand,
        provider="auto"
    )


# ============================================
# UTILITY FUNCTIONS (for compatibility)
# ============================================

def normalize_brand(text: str) -> str:
    """Clean and normalize brand text."""
    if not text:
        return ""
    return " ".join(text.split()).strip(".,!?;:'\"()[]{}…")


def is_known_brand(text: str) -> bool:
    """
    Check if text is a known brand.
    With LLM approach, we don't maintain a static list.
    """
    return False  # Let the LLM decide


def add_known_brand(brand: str) -> None:
    """No-op for compatibility. LLM approach doesn't need this."""
    pass


def add_known_brands(brands: List[str]) -> None:
    """No-op for compatibility. LLM approach doesn't need this."""
    pass


def get_known_brands() -> Set[str]:
    """Return empty set. LLM approach doesn't maintain a static list."""
    return set()


def is_definitely_not_brand(word: str) -> bool:
    """
    For compatibility. LLM approach handles this internally.
    """
    return False
