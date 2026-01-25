# query_generator.py
"""
AI-powered query generation for GEO Tracker.
Uses LLM APIs to generate industry-specific, relevant queries based on business context.

API Keys should be set in environment variables:
- OPENAI_API_KEY for OpenAI
- GOOGLE_API_KEY for Google Gemini
"""
import os
import json
import re
from typing import List, Dict, Optional
from dataclasses import dataclass

# Try to import OpenAI
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Try to import Google Generative AI
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


@dataclass
class BusinessContext:
    """Business context for query generation."""
    company_name: str
    industry: str
    description: Optional[str] = None
    target_market: Optional[str] = None
    language: str = "en"
    focus_areas: Optional[List[str]] = None
    competitor_names: Optional[List[str]] = None


# Query categories for classification
QUERY_CATEGORIES = [
    "product_recommendation",
    "brand_comparison", 
    "best_practices",
    "buying_guide",
    "expert_advice",
    "review_request",
    "problem_solving",
    "market_research",
    "general_inquiry",
]

# Language-specific prompt templates
PROMPT_TEMPLATES = {
    "en": """You are an expert market researcher specializing in {industry}.

IMPORTANT CONTEXT:
- Target Market/Country: {target_market}
- All queries MUST be relevant to consumers in {target_market}
- Company being analyzed: {company_name} (DO NOT mention this name in queries!)
- Business: {description}

Your task: Generate {count} realistic search queries that a consumer in {target_market} might ask when researching products/services in the {industry} industry.

CRITICAL RULES:
1. DO NOT include "{company_name}" in ANY query - we are testing ORGANIC visibility (whether AI mentions the brand without being asked about it specifically)
2. ALL queries must be relevant to {target_market} market (use local context, currency, regulations, culture)
3. Generate queries that real consumers would actually search for
4. Include a mix of:
   - General product/service questions ("best X in {target_market}", "top X recommendations")
   - Comparison queries ("X vs Y", "which is better")
   - Problem-solving queries ("how to choose", "what to look for")
   - Review/recommendation queries ("recommended X", "trusted X")
   - Local market queries (specific to {target_market})

Focus Areas: {focus_areas}
Known Competitors: {competitors}

Output Format:
Return a JSON array where each item has:
- "question": the query text (NEVER include {company_name}!)
- "category": one of {categories}
- "intent": brief description of what the searcher wants

Return ONLY the JSON array, no other text.""",

    "de": """Du bist ein Experte für Marktforschung im Bereich {industry}.

WICHTIGER KONTEXT:
- Zielmarkt/Land: {target_market}
- Alle Anfragen MÜSSEN für Verbraucher in {target_market} relevant sein
- Analysiertes Unternehmen: {company_name} (Diesen Namen NICHT in Anfragen erwähnen!)
- Geschäft: {description}

Deine Aufgabe: Generiere {count} realistische Suchanfragen, die ein Verbraucher in {target_market} bei der Recherche nach Produkten/Dienstleistungen in der {industry}-Branche stellen könnte.

KRITISCHE REGELN:
1. "{company_name}" NICHT in Anfragen erwähnen - wir testen ORGANISCHE Sichtbarkeit (ob KI die Marke ohne direkte Frage erwähnt)
2. ALLE Anfragen müssen für den {target_market}-Markt relevant sein
3. Generiere Anfragen, die echte Verbraucher tatsächlich suchen würden
4. Mische verschiedene Typen:
   - Allgemeine Produkt-/Servicefragen ("beste X in {target_market}", "Top X Empfehlungen")
   - Vergleichsanfragen ("X vs Y", "was ist besser")
   - Problemlösungs-Anfragen ("wie wählen", "worauf achten")
   - Bewertungs-/Empfehlungsanfragen ("empfohlene X", "vertrauenswürdige X")
   - Lokale Marktanfragen (spezifisch für {target_market})

Schwerpunkte: {focus_areas}
Bekannte Wettbewerber: {competitors}

Ausgabeformat:
Gib ein JSON-Array zurück, wobei jedes Element hat:
- "question": der Anfragetext (NIEMALS {company_name} erwähnen!)
- "category": einer von {categories}
- "intent": kurze Beschreibung, was der Suchende möchte

Gib NUR das JSON-Array zurück, keinen anderen Text.""",

    "fr": """Vous êtes un expert en études de marché spécialisé dans {industry}.

CONTEXTE IMPORTANT:
- Marché cible/Pays: {target_market}
- Toutes les requêtes DOIVENT être pertinentes pour les consommateurs en {target_market}
- Entreprise analysée: {company_name} (NE PAS mentionner ce nom dans les requêtes!)
- Activité: {description}

Votre tâche: Générez {count} requêtes de recherche réalistes qu'un consommateur en {target_market} pourrait poser lors de recherches sur les produits/services dans l'industrie {industry}.

RÈGLES CRITIQUES:
1. NE PAS inclure "{company_name}" dans AUCUNE requête - nous testons la visibilité ORGANIQUE
2. TOUTES les requêtes doivent être pertinentes pour le marché {target_market}
3. Générez des requêtes que les vrais consommateurs rechercheraient
4. Incluez un mélange de types de requêtes locales et générales

Domaines d'intérêt: {focus_areas}
Concurrents connus: {competitors}

Format de sortie:
Retournez un tableau JSON où chaque élément a:
- "question": le texte de la requête (JAMAIS {company_name}!)
- "category": une des catégories {categories}
- "intent": brève description

Retournez UNIQUEMENT le tableau JSON.""",

    "es": """Eres un experto en investigación de mercado especializado en {industry}.

CONTEXTO IMPORTANTE:
- Mercado objetivo/País: {target_market}
- Todas las consultas DEBEN ser relevantes para consumidores en {target_market}
- Empresa analizada: {company_name} (¡NO mencionar este nombre en las consultas!)
- Negocio: {description}

Tu tarea: Genera {count} consultas de búsqueda realistas que un consumidor en {target_market} podría hacer al investigar productos/servicios en la industria de {industry}.

REGLAS CRÍTICAS:
1. NO incluir "{company_name}" en NINGUNA consulta - estamos probando visibilidad ORGÁNICA
2. TODAS las consultas deben ser relevantes para el mercado de {target_market}
3. Genera consultas que los consumidores reales buscarían
4. Incluye una mezcla de tipos de consultas locales y generales

Áreas de enfoque: {focus_areas}
Competidores conocidos: {competitors}

Formato de salida:
Devuelve un array JSON donde cada elemento tiene:
- "question": el texto de la consulta (¡NUNCA {company_name}!)
- "category": una de las categorías {categories}
- "intent": breve descripción

Devuelve SOLO el array JSON.""",

    "it": """Sei un esperto di ricerche di mercato specializzato in {industry}.

CONTESTO IMPORTANTE:
- Mercato target/Paese: {target_market}
- Tutte le query DEVONO essere rilevanti per i consumatori in {target_market}
- Azienda analizzata: {company_name} (NON menzionare questo nome nelle query!)
- Attività: {description}

Il tuo compito: Genera {count} query di ricerca realistiche che un consumatore in {target_market} potrebbe fare quando cerca prodotti/servizi nel settore {industry}.

REGOLE CRITICHE:
1. NON includere "{company_name}" in NESSUNA query - stiamo testando la visibilità ORGANICA
2. TUTTE le query devono essere rilevanti per il mercato {target_market}
3. Genera query che i veri consumatori cercherebbero
4. Includi un mix di tipi di query locali e generali

Aree di focus: {focus_areas}
Concorrenti noti: {competitors}

Formato output:
Restituisci un array JSON dove ogni elemento ha:
- "question": il testo della query (MAI {company_name}!)
- "category": una delle categorie {categories}
- "intent": breve descrizione

Restituisci SOLO l'array JSON.""",
}


def _parse_json_response(text: str) -> List[Dict]:
    """Parse JSON from LLM response, handling markdown code blocks."""
    # Remove markdown code blocks if present
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    # Try to find JSON array in the text
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        text = match.group(0)
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # If parsing fails, return empty list
        return []


def generate_queries_with_openai(
    context: BusinessContext,
    count: int = 25,
    model: str = "gpt-4.1-mini"
) -> List[Dict]:
    """
    Generate queries using OpenAI API.
    
    Args:
        context: Business context for query generation
        count: Number of queries to generate
        model: OpenAI model to use
    
    Returns:
        List of query dictionaries with question, category, intent
    """
    if not HAS_OPENAI:
        raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=api_key)
    
    # Select language template
    lang = context.language.lower()[:2] if context.language else "en"
    template = PROMPT_TEMPLATES.get(lang, PROMPT_TEMPLATES["en"])
    
    # Build prompt
    prompt = template.format(
        industry=context.industry,
        company_name=context.company_name,
        description=context.description or f"A company in the {context.industry} industry",
        target_market=context.target_market or "General consumers",
        focus_areas=", ".join(context.focus_areas) if context.focus_areas else "All product areas",
        competitors=", ".join(context.competitor_names) if context.competitor_names else "Unknown",
        count=count,
        categories=", ".join(QUERY_CATEGORIES)
    )
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a market research expert. Always respond with valid JSON only."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=4000,
        )
        
        result_text = response.choices[0].message.content
        queries = _parse_json_response(result_text)
        
        # Add prompt_id to each query
        for i, q in enumerate(queries):
            q["prompt_id"] = f"ai_gen_{i+1}"
            # Ensure required fields exist
            if "question" not in q:
                continue
            if "category" not in q:
                q["category"] = "general_inquiry"
        
        # Filter out invalid queries
        queries = [q for q in queries if q.get("question")]
        
        return queries[:count]
        
    except Exception as e:
        print(f"[error] OpenAI query generation failed: {e}")
        raise


def generate_queries_with_gemini(
    context: BusinessContext,
    count: int = 25,
    model: str = "gemini-2.5-flash"
) -> List[Dict]:
    """
    Generate queries using Google Gemini API.
    
    Args:
        context: Business context for query generation
        count: Number of queries to generate
        model: Gemini model to use
    
    Returns:
        List of query dictionaries with question, category, intent
    """
    if not HAS_GEMINI:
        raise ImportError("Google Generative AI package not installed. Run: pip install google-generativeai")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    genai.configure(api_key=api_key)
    
    # Select language template
    lang = context.language.lower()[:2] if context.language else "en"
    template = PROMPT_TEMPLATES.get(lang, PROMPT_TEMPLATES["en"])
    
    # Build prompt
    prompt = template.format(
        industry=context.industry,
        company_name=context.company_name,
        description=context.description or f"A company in the {context.industry} industry",
        target_market=context.target_market or "General consumers",
        focus_areas=", ".join(context.focus_areas) if context.focus_areas else "All product areas",
        competitors=", ".join(context.competitor_names) if context.competitor_names else "Unknown",
        count=count,
        categories=", ".join(QUERY_CATEGORIES)
    )
    
    try:
        gemini_model = genai.GenerativeModel(model)
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.8,
                max_output_tokens=4000,
            )
        )
        
        result_text = response.text
        queries = _parse_json_response(result_text)
        
        # Add prompt_id to each query
        for i, q in enumerate(queries):
            q["prompt_id"] = f"ai_gen_{i+1}"
            # Ensure required fields exist
            if "question" not in q:
                continue
            if "category" not in q:
                q["category"] = "general_inquiry"
        
        # Filter out invalid queries
        queries = [q for q in queries if q.get("question")]
        
        return queries[:count]
        
    except Exception as e:
        print(f"[error] Gemini query generation failed: {e}")
        raise


def generate_queries(
    context: BusinessContext,
    count: int = 25,
    provider: str = "auto",
    model: Optional[str] = None
) -> List[Dict]:
    """
    Generate queries using the best available LLM provider.
    
    Args:
        context: Business context for query generation
        count: Number of queries to generate
        provider: "openai", "gemini", or "auto" (tries OpenAI first, then Gemini)
        model: Optional model override
    
    Returns:
        List of query dictionaries
    """
    if provider == "auto":
        # Try OpenAI first, then Gemini
        if os.getenv("OPENAI_API_KEY") and HAS_OPENAI:
            provider = "openai"
        elif os.getenv("GOOGLE_API_KEY") and HAS_GEMINI:
            provider = "gemini"
        else:
            raise ValueError(
                "No LLM API key configured. Set OPENAI_API_KEY or GOOGLE_API_KEY environment variable."
            )
    
    if provider == "openai":
        return generate_queries_with_openai(
            context, 
            count, 
            model or "gpt-4.1-mini"
        )
    elif provider == "gemini":
        return generate_queries_with_gemini(
            context, 
            count, 
            model or "gemini-2.5-flash"
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


# Fallback sample queries when API is not available
def get_fallback_queries(
    context: BusinessContext,
    count: int = 25
) -> List[Dict]:
    """
    Return sample queries when LLM API is not available.
    These are generated based on the business context.
    """
    company = context.company_name
    industry = context.industry
    
    # Language-specific query templates
    if context.language and context.language.lower().startswith("de"):
        templates = [
            f"Was sind die besten {industry} Marken in Deutschland?",
            f"Vergleich {industry} Produkte 2024",
            f"Wo kann ich {industry} Produkte online kaufen?",
            f"Worauf sollte ich beim Kauf von {industry} achten?",
            f"Beste {industry} für Anfänger",
            f"Top bewertete {industry} Marken",
            f"Natürliche {industry} Empfehlungen",
            f"Bio {industry} Marken Deutschland",
            f"{industry} Qualitätsvergleich",
            f"Beste Preis-Leistung {industry}",
            f"Premium {industry} Marken Test",
            f"Welche {industry} Marke ist am vertrauenswürdigsten?",
            f"Kundenbewertungen {industry}",
            f"Beste {industry} für Gesundheit",
            f"Was ist besser: Marke A oder Marke B {industry}?",
            f"Nachhaltige {industry} Produkte",
            f"Vegane {industry} Alternativen",
            f"Günstige {industry} Empfehlungen",
            f"{industry} ohne Zusatzstoffe",
            f"Hochwertige {industry} kaufen",
            f"Empfehlenswerte {industry} Anbieter",
            f"Test {industry} Produkte Stiftung Warentest",
            f"Welche {industry} sind am besten bewertet?",
            f"Erfahrungen mit {industry} Produkten",
            f"Gibt es gute deutsche {industry} Hersteller?",
        ]
    else:
        templates = [
            f"What are the best {industry} brands?",
            f"Compare {industry} products 2024",
            f"Where can I buy {industry} products online?",
            f"What should I look for when buying {industry}?",
            f"Best {industry} for beginners",
            f"Top rated {industry} brands",
            f"Natural {industry} recommendations",
            f"Organic {industry} brands",
            f"{industry} quality comparison",
            f"Best value {industry} products",
            f"Premium {industry} brands review",
            f"Which {industry} brand is most trustworthy?",
            f"Customer reviews for {industry}",
            f"Best {industry} for health benefits",
            f"What's better: Brand A or Brand B {industry}?",
            f"Sustainable {industry} products",
            f"Vegan {industry} alternatives",
            f"Budget-friendly {industry} recommendations",
            f"{industry} without additives",
            f"High-quality {industry} to buy",
            f"Recommended {industry} providers",
            f"Expert reviews {industry} products",
            f"Which {industry} are best rated?",
            f"Experiences with {industry} products",
            f"Are there good local {industry} manufacturers?",
        ]
    
    # Add some brand-specific queries (2-3 only)
    brand_queries = [
        f"Is {company} a good {industry} brand?",
        f"{company} reviews",
        f"{company} vs competitors",
    ]
    
    # Combine and select
    all_queries = templates + brand_queries
    
    queries = []
    for i, q in enumerate(all_queries[:count]):
        category = QUERY_CATEGORIES[i % len(QUERY_CATEGORIES)]
        queries.append({
            "question": q,
            "category": category,
            "prompt_id": f"sample_{i+1}",
            "intent": f"User researching {industry}",
            "generated_by": "fallback"
        })
    
    return queries
