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
# HELPER FUNCTIONS
# ============================================

def _filter_our_brand_variations(brands: List, our_brand: str) -> Set[str]:
    """
    Filter out our brand and its variations from the detected brands.

    Handles cases like:
    - "Sunday Natural" vs "Sunday" vs "sunday.de"
    - Exact matches
    - Domain-style variations

    IMPORTANT: For multi-word brands like "Sunday Natural":
    - Filter "Sunday" (the distinctive/unique first word)
    - DON'T filter "Natural" (generic word that could be part of competitor names)
    """
    if not our_brand:
        return {b.strip() for b in brands if isinstance(b, str) and b.strip()}

    our_brand_lower = our_brand.lower().strip()
    our_brand_words = our_brand_lower.split()

    # For multi-word brands, only the FIRST word is considered distinctive
    # e.g., "Sunday Natural" -> "Sunday" is distinctive, "Natural" is generic
    # e.g., "Nature Love" -> "Nature" could be distinctive but "Love" is generic
    distinctive_word = our_brand_words[0] if our_brand_words else ""

    # Generic words that should NOT be filtered even if they appear in our brand
    # These are common words that many brands use
    GENERIC_BRAND_WORDS = {
        "natural", "nature", "organic", "bio", "pure", "health", "healthy",
        "life", "love", "care", "plus", "pro", "premium", "gold", "best",
        "super", "ultra", "max", "active", "vital", "fit", "wellness",
        "green", "eco", "fresh", "original", "classic", "elements", "essentials"
    }

    # Domain-style variations (e.g., "sunday.de", "sundaynatural.com")
    our_brand_no_spaces = our_brand_lower.replace(" ", "")

    cleaned = set()
    for b in brands:
        if not isinstance(b, str) or not b.strip():
            continue

        b_original = b.strip()
        b_lower = b_original.lower()
        b_no_spaces = b_lower.replace(" ", "").replace("-", "").replace(".", "").replace("_", "")

        # Check if this brand should be filtered out
        should_filter = False
        filter_reason = ""

        # 1. Exact match of full brand name
        if b_lower == our_brand_lower:
            should_filter = True
            filter_reason = "exact match"

        # 2. Detected brand contains our full brand name
        elif our_brand_lower in b_lower:
            should_filter = True
            filter_reason = "contains full brand"

        # 3. Our full brand contains the detected brand AND it's the distinctive word
        elif b_lower == distinctive_word:
            should_filter = True
            filter_reason = "distinctive word match"

        # 4. Domain-style variations
        else:
            b_base = re.sub(r'\.(com|de|co|net|org|io|uk|eu|fr|it|es)$', '', b_lower)

            # Check if domain matches our brand (e.g., "sundaynatural.de" or "sunday.de")
            if b_base == our_brand_no_spaces:
                should_filter = True
                filter_reason = "domain matches full brand"
            elif b_base == distinctive_word:
                should_filter = True
                filter_reason = "domain matches distinctive word"
            # Check if detected brand starts with our distinctive word + domain
            elif distinctive_word and b_lower.startswith(distinctive_word) and b_lower != distinctive_word:
                # Only filter if it looks like a variation of our brand
                # e.g., "sundaynatural.com" for "Sunday Natural"
                # but NOT "Sundance" or "Sunflower"
                if b_no_spaces.startswith(our_brand_no_spaces) or our_brand_no_spaces.startswith(b_no_spaces):
                    should_filter = True
                    filter_reason = "brand name variation"

        if should_filter:
            print(f"[brand_detection] Filtered out '{b_original}' ({filter_reason} of '{our_brand}')")
        else:
            cleaned.add(b_original)

    return cleaned


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

        # Build exclusion hint - only exclude distinctive first word, not generic words
        our_brand_words = our_brand.split() if our_brand else []
        distinctive_word = our_brand_words[0] if our_brand_words else ""
        our_brand_variations_hint = ""
        if our_brand and distinctive_word:
            our_brand_variations_hint = f"""
   - CRITICAL: Exclude "{our_brand}" and shortened forms using "{distinctive_word}"
   - Example: For "Sunday Natural", exclude "Sunday Natural", "Sunday", "sunday.de"
   - BUT DO include competitor brands with generic words like "Natural Elements", "Nature Love", etc."""

        prompt = f"""Extract ONLY actual COMPETITOR company/brand names from the following text.

CONTEXT:
- Industry: {industry}
- Market/Country: {market}
- OUR brand to EXCLUDE: "{our_brand}"

RULES:
1. Return ONLY real company names and brand names that are COMPETITORS
2. DO NOT include:
   - Country names (India, Germany, USA, etc.)
   - City names (Berlin, Mumbai, Delhi, etc.)
   - Generic words (Food, Delivery, Restaurant, Quality, etc.)
   - Adjectives (Best, Top, Popular, Indian, German, etc.){our_brand_variations_hint}
3. Include competitor brands only (not our brand or its variations)
4. IMPORTANT - Brand variations and aliases:
   - Recognize that brands often appear in multiple forms: full name, shortened name, domain name, etc.
   - Examples of variations that are the SAME brand:
     * "Amazon", "amazon.de", "Amazon.com" → use "Amazon"
     * "dm-drogerie markt", "dm", "dm.de" → use "dm"
     * "Natural Elements", "naturalelements.de" → use "Natural Elements"
   - Consolidate variations into ONE canonical (most complete/official) form
5. Return as JSON array of strings with deduplicated canonical names only

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
            cleaned = _filter_our_brand_variations(brands, our_brand)
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

        # Build exclusion hint - only exclude distinctive first word, not generic words
        our_brand_words = our_brand.split() if our_brand else []
        distinctive_word = our_brand_words[0] if our_brand_words else ""
        our_brand_variations_hint = ""
        if our_brand and distinctive_word:
            our_brand_variations_hint = f"""
   - CRITICAL: Exclude "{our_brand}" and shortened forms using "{distinctive_word}"
   - Example: For "Sunday Natural", exclude "Sunday Natural", "Sunday", "sunday.de"
   - BUT DO include competitor brands with generic words like "Natural Elements", "Nature Love", etc."""

        prompt = f"""Extract ONLY actual COMPETITOR company/brand names from the following text.

CONTEXT:
- Industry: {industry}
- Market/Country: {market}
- OUR brand to EXCLUDE: "{our_brand}"

RULES:
1. Return ONLY real company names and brand names that are COMPETITORS
2. DO NOT include:
   - Country names (India, Germany, USA, etc.)
   - City names (Berlin, Mumbai, Delhi, etc.)
   - Generic words (Food, Delivery, Restaurant, Quality, etc.)
   - Adjectives (Best, Top, Popular, Indian, German, etc.){our_brand_variations_hint}
3. Include competitor brands only (not our brand or its variations)
4. IMPORTANT - Brand variations and aliases:
   - Recognize that brands often appear in multiple forms: full name, shortened name, domain name, etc.
   - Examples of variations that are the SAME brand:
     * "Amazon", "amazon.de", "Amazon.com" → use "Amazon"
     * "dm-drogerie markt", "dm", "dm.de" → use "dm"
     * "Natural Elements", "naturalelements.de" → use "Natural Elements"
   - Consolidate variations into ONE canonical (most complete/official) form
5. Return as JSON array of strings with deduplicated canonical names only

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
            # Filter out our brand and clean up
            cleaned = _filter_our_brand_variations(brands, our_brand)
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
