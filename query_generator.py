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
from datetime import datetime
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


# Query categories for classification (funnel-aligned)
QUERY_CATEGORIES = [
    "shopping_intent",
    "research_comparison",
    "problem_solution",
    "trust_reviews",
]

# Language names for display
LANGUAGE_NAMES = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
}

# Language-specific prompt templates
PROMPT_TEMPLATES = {
    "en": """You are an expert market researcher specializing in {industry}.

CONTEXT:
- Target Market/Country: {target_market}
- Company being analyzed: {company_name} (DO NOT mention this name in queries!)
- Business Description: {description}
- Current Year: {current_year}
- Language: English

YOUR TASK: Generate {count} realistic queries that consumers in {target_market} would ask AI assistants (ChatGPT, Claude, Gemini, Perplexity) when researching or buying {industry} products/services.

CRITICAL RULES:
1. NEVER include "{company_name}" in any query - we test ORGANIC visibility (whether AI mentions the brand without being asked about it specifically)
2. ALL queries must be culturally and contextually relevant to {target_market} (local language nuances, regulations, currency, shopping habits, local preferences)
3. When referencing time, use "{current_year}" or "this year" - never hardcode past years
4. Queries should reflect how real people ask AI assistants (conversational, natural language)
5. Include LOCAL MARKET specific queries (e.g., "{target_market} regulations", "buy in {target_market}", "shipped to {target_market}", local retailer references)

QUERY DISTRIBUTION - Generate queries in this approximate mix:

**Shopping/Purchase Intent (40% of queries):**
- "Best [product] to buy in {target_market}"
- "Where can I buy [product] online in {target_market}"
- "[Product A] vs [Product B] - which should I buy"
- "Is [product] worth the price"
- "Best value [product] in {current_year}"
- "Affordable [product] recommendations"
- "Premium [product] worth buying"
- "Best [product] deals in {target_market}"

**Research/Comparison (25% of queries):**
- "Best [product] brands in {target_market} {current_year}"
- "Top rated [product] this year"
- "Compare [product] options"
- "What's the difference between [type A] and [type B]"
- "[Competitor] alternatives"
- "Best [product] brands available in {target_market}"

**Problem/Solution (20% of queries):**
- "What [product] helps with [problem]"
- "Best [product] for [specific need/condition]"
- "How to choose the right [product]"
- "What to look for when buying [product]"
- "[Problem] - what [product] do experts recommend"

**Trust/Reviews (15% of queries):**
- "Most trusted [product] brands in {target_market}"
- "Are [product type] from {target_market} good quality"
- "Reliable [product] brands with good reviews"
- "Which [product] brands have best reputation in {target_market}"
- "[Product] brands certified in {target_market}"

FOCUS AREAS: {focus_areas}
KNOWN COMPETITORS (use these in comparison queries where relevant): {competitors}

OUTPUT FORMAT:
Return a JSON array where each item has:
- "question": the query text (NEVER include {company_name}!)
- "category": one of {categories}
- "intent": brief description of searcher's goal
- "funnel_stage": one of ["bottom", "middle", "top"]

IMPORTANT: All {count} queries must be in English and relevant to {target_market} market.

Return ONLY valid JSON array, no other text.""",

    "de": """Du bist ein Experte für Marktforschung im Bereich {industry}.

KONTEXT:
- Zielmarkt/Land: {target_market}
- Analysiertes Unternehmen: {company_name} (Diesen Namen NICHT in Anfragen erwähnen!)
- Geschäftsbeschreibung: {description}
- Aktuelles Jahr: {current_year}
- Sprache: Deutsch

DEINE AUFGABE: Generiere {count} realistische Anfragen, die Verbraucher in {target_market} an KI-Assistenten (ChatGPT, Claude, Gemini, Perplexity) stellen würden, wenn sie {industry} Produkte/Dienstleistungen recherchieren oder kaufen.

KRITISCHE REGELN:
1. "{company_name}" NIEMALS in einer Anfrage erwähnen - wir testen ORGANISCHE Sichtbarkeit (ob KI die Marke ohne direkte Frage erwähnt)
2. ALLE Anfragen müssen kulturell und kontextuell relevant für {target_market} sein (lokale Sprachnuancen, Vorschriften, Währung, Einkaufsgewohnheiten, lokale Präferenzen)
3. Bei Zeitbezügen "{current_year}" oder "dieses Jahr" verwenden - niemals vergangene Jahre fest codieren
4. Anfragen sollten widerspiegeln, wie echte Menschen KI-Assistenten fragen (umgangssprachlich, natürliche Sprache)
5. LOKALE MARKT-spezifische Anfragen einbeziehen (z.B. "{target_market} Vorschriften", "in {target_market} kaufen", "Lieferung nach {target_market}", lokale Händlerreferenzen)

ANFRAGENVERTEILUNG - Generiere Anfragen in dieser ungefähren Mischung:

**Kaufabsicht (40% der Anfragen):**
- "Beste [Produkt] zum Kaufen in {target_market}"
- "Wo kann ich [Produkt] online in {target_market} kaufen"
- "[Produkt A] vs [Produkt B] - was soll ich kaufen"
- "Lohnt sich [Produkt] preislich"
- "Bestes Preis-Leistungs-Verhältnis [Produkt] {current_year}"
- "Günstige [Produkt] Empfehlungen"
- "Premium [Produkt] kaufenswert"
- "Beste [Produkt] Angebote in {target_market}"

**Recherche/Vergleich (25% der Anfragen):**
- "Beste [Produkt] Marken in {target_market} {current_year}"
- "Top bewertete [Produkt] dieses Jahr"
- "[Produkt] Optionen vergleichen"
- "Was ist der Unterschied zwischen [Typ A] und [Typ B]"
- "[Wettbewerber] Alternativen"
- "Beste [Produkt] Marken verfügbar in {target_market}"

**Problem/Lösung (20% der Anfragen):**
- "Welches [Produkt] hilft bei [Problem]"
- "Bestes [Produkt] für [spezifischer Bedarf/Zustand]"
- "Wie wähle ich das richtige [Produkt]"
- "Worauf achten beim [Produkt] Kauf"
- "[Problem] - welches [Produkt] empfehlen Experten"

**Vertrauen/Bewertungen (15% der Anfragen):**
- "Vertrauenswürdigste [Produkt] Marken in {target_market}"
- "Sind [Produkttyp] aus {target_market} qualitativ gut"
- "Zuverlässige [Produkt] Marken mit guten Bewertungen"
- "Welche [Produkt] Marken haben den besten Ruf in {target_market}"
- "[Produkt] Marken zertifiziert in {target_market}"

SCHWERPUNKTE: {focus_areas}
BEKANNTE WETTBEWERBER (in Vergleichsanfragen verwenden): {competitors}

AUSGABEFORMAT:
Gib ein JSON-Array zurück, wobei jedes Element hat:
- "question": der Anfragetext (NIEMALS {company_name}!)
- "category": einer von {categories}
- "intent": kurze Beschreibung des Suchziels
- "funnel_stage": einer von ["bottom", "middle", "top"]

WICHTIG: Alle {count} Anfragen müssen auf Deutsch sein und relevant für den {target_market} Markt.

Gib NUR ein gültiges JSON-Array zurück, keinen anderen Text.""",

    "fr": """Vous êtes un expert en études de marché spécialisé dans {industry}.

CONTEXTE:
- Marché cible/Pays: {target_market}
- Entreprise analysée: {company_name} (NE PAS mentionner ce nom dans les requêtes!)
- Description de l'activité: {description}
- Année en cours: {current_year}
- Langue: Français

VOTRE TÂCHE: Générez {count} requêtes réalistes que les consommateurs de {target_market} poseraient aux assistants IA (ChatGPT, Claude, Gemini, Perplexity) lors de recherches ou d'achats de produits/services {industry}.

RÈGLES CRITIQUES:
1. NE JAMAIS inclure "{company_name}" dans aucune requête - nous testons la visibilité ORGANIQUE (si l'IA mentionne la marque sans qu'on le lui demande spécifiquement)
2. TOUTES les requêtes doivent être culturellement et contextuellement pertinentes pour {target_market} (nuances linguistiques locales, réglementations, devise, habitudes d'achat, préférences locales)
3. Pour les références temporelles, utilisez "{current_year}" ou "cette année" - ne jamais coder en dur des années passées
4. Les requêtes doivent refléter la façon dont les vraies personnes interrogent les assistants IA (conversationnel, langage naturel)
5. Incluez des requêtes spécifiques au MARCHÉ LOCAL (ex: "réglementations {target_market}", "acheter en {target_market}", "livraison vers {target_market}", références aux détaillants locaux)

DISTRIBUTION DES REQUÊTES - Générez des requêtes dans ce mélange approximatif:

**Intention d'achat (40% des requêtes):**
- "Meilleur [produit] à acheter en {target_market}"
- "Où acheter [produit] en ligne en {target_market}"
- "[Produit A] vs [Produit B] - lequel acheter"
- "[Produit] vaut-il son prix"
- "Meilleur rapport qualité-prix [produit] {current_year}"
- "Recommandations [produit] abordables"
- "[Produit] premium à acheter"
- "Meilleures offres [produit] en {target_market}"

**Recherche/Comparaison (25% des requêtes):**
- "Meilleures marques de [produit] en {target_market} {current_year}"
- "[Produit] les mieux notés cette année"
- "Comparer les options de [produit]"
- "Quelle est la différence entre [type A] et [type B]"
- "Alternatives à [concurrent]"
- "Meilleures marques de [produit] disponibles en {target_market}"

**Problème/Solution (20% des requêtes):**
- "Quel [produit] aide pour [problème]"
- "Meilleur [produit] pour [besoin spécifique/condition]"
- "Comment choisir le bon [produit]"
- "Que rechercher lors de l'achat de [produit]"
- "[Problème] - quel [produit] recommandent les experts"

**Confiance/Avis (15% des requêtes):**
- "Marques de [produit] les plus fiables en {target_market}"
- "Les [type de produit] de {target_market} sont-ils de bonne qualité"
- "Marques de [produit] fiables avec de bons avis"
- "Quelles marques de [produit] ont la meilleure réputation en {target_market}"
- "Marques de [produit] certifiées en {target_market}"

DOMAINES D'INTÉRÊT: {focus_areas}
CONCURRENTS CONNUS (utiliser dans les requêtes de comparaison): {competitors}

FORMAT DE SORTIE:
Retournez un tableau JSON où chaque élément a:
- "question": le texte de la requête (JAMAIS {company_name}!)
- "category": un parmi {categories}
- "intent": brève description de l'objectif du chercheur
- "funnel_stage": un parmi ["bottom", "middle", "top"]

IMPORTANT: Les {count} requêtes doivent être en français et pertinentes pour le marché {target_market}.

Retournez UNIQUEMENT un tableau JSON valide, aucun autre texte.""",

    "es": """Eres un experto en investigación de mercado especializado en {industry}.

CONTEXTO:
- Mercado objetivo/País: {target_market}
- Empresa analizada: {company_name} (¡NO mencionar este nombre en las consultas!)
- Descripción del negocio: {description}
- Año actual: {current_year}
- Idioma: Español

TU TAREA: Genera {count} consultas realistas que los consumidores de {target_market} harían a asistentes de IA (ChatGPT, Claude, Gemini, Perplexity) al investigar o comprar productos/servicios de {industry}.

REGLAS CRÍTICAS:
1. NUNCA incluir "{company_name}" en ninguna consulta - probamos la visibilidad ORGÁNICA (si la IA menciona la marca sin que se le pregunte específicamente)
2. TODAS las consultas deben ser cultural y contextualmente relevantes para {target_market} (matices del idioma local, regulaciones, moneda, hábitos de compra, preferencias locales)
3. Al hacer referencias temporales, usa "{current_year}" o "este año" - nunca codificar años pasados
4. Las consultas deben reflejar cómo las personas reales preguntan a los asistentes de IA (conversacional, lenguaje natural)
5. Incluye consultas específicas del MERCADO LOCAL (ej: "regulaciones de {target_market}", "comprar en {target_market}", "envío a {target_market}", referencias a minoristas locales)

DISTRIBUCIÓN DE CONSULTAS - Genera consultas en esta mezcla aproximada:

**Intención de compra (40% de las consultas):**
- "Mejor [producto] para comprar en {target_market}"
- "Dónde comprar [producto] online en {target_market}"
- "[Producto A] vs [Producto B] - cuál debo comprar"
- "¿Vale la pena el precio de [producto]"
- "Mejor relación calidad-precio [producto] {current_year}"
- "Recomendaciones de [producto] asequibles"
- "[Producto] premium que vale la pena comprar"
- "Mejores ofertas de [producto] en {target_market}"

**Investigación/Comparación (25% de las consultas):**
- "Mejores marcas de [producto] en {target_market} {current_year}"
- "[Producto] mejor valorados este año"
- "Comparar opciones de [producto]"
- "Cuál es la diferencia entre [tipo A] y [tipo B]"
- "Alternativas a [competidor]"
- "Mejores marcas de [producto] disponibles en {target_market}"

**Problema/Solución (20% de las consultas):**
- "Qué [producto] ayuda con [problema]"
- "Mejor [producto] para [necesidad específica/condición]"
- "Cómo elegir el [producto] correcto"
- "Qué buscar al comprar [producto]"
- "[Problema] - qué [producto] recomiendan los expertos"

**Confianza/Reseñas (15% de las consultas):**
- "Marcas de [producto] más confiables en {target_market}"
- "¿Son de buena calidad los [tipo de producto] de {target_market}"
- "Marcas de [producto] confiables con buenas reseñas"
- "Qué marcas de [producto] tienen mejor reputación en {target_market}"
- "Marcas de [producto] certificadas en {target_market}"

ÁREAS DE ENFOQUE: {focus_areas}
COMPETIDORES CONOCIDOS (usar en consultas de comparación): {competitors}

FORMATO DE SALIDA:
Devuelve un array JSON donde cada elemento tiene:
- "question": el texto de la consulta (¡NUNCA {company_name}!)
- "category": uno de {categories}
- "intent": breve descripción del objetivo del buscador
- "funnel_stage": uno de ["bottom", "middle", "top"]

IMPORTANTE: Las {count} consultas deben estar en español y ser relevantes para el mercado de {target_market}.

Devuelve SOLO un array JSON válido, ningún otro texto.""",

    "it": """Sei un esperto di ricerche di mercato specializzato in {industry}.

CONTESTO:
- Mercato target/Paese: {target_market}
- Azienda analizzata: {company_name} (NON menzionare questo nome nelle query!)
- Descrizione dell'attività: {description}
- Anno corrente: {current_year}
- Lingua: Italiano

IL TUO COMPITO: Genera {count} query realistiche che i consumatori in {target_market} farebbero agli assistenti IA (ChatGPT, Claude, Gemini, Perplexity) quando ricercano o acquistano prodotti/servizi {industry}.

REGOLE CRITICHE:
1. MAI includere "{company_name}" in nessuna query - testiamo la visibilità ORGANICA (se l'IA menziona il brand senza che venga chiesto specificamente)
2. TUTTE le query devono essere culturalmente e contestualmente rilevanti per {target_market} (sfumature linguistiche locali, regolamenti, valuta, abitudini di acquisto, preferenze locali)
3. Per riferimenti temporali, usa "{current_year}" o "quest'anno" - mai codificare anni passati
4. Le query devono riflettere come le persone reali chiedono agli assistenti IA (conversazionale, linguaggio naturale)
5. Includi query specifiche per il MERCATO LOCALE (es: "regolamenti {target_market}", "comprare in {target_market}", "spedizione verso {target_market}", riferimenti a rivenditori locali)

DISTRIBUZIONE DELLE QUERY - Genera query in questo mix approssimativo:

**Intenzione d'acquisto (40% delle query):**
- "Miglior [prodotto] da comprare in {target_market}"
- "Dove comprare [prodotto] online in {target_market}"
- "[Prodotto A] vs [Prodotto B] - quale dovrei comprare"
- "[Prodotto] vale il prezzo"
- "Miglior rapporto qualità-prezzo [prodotto] {current_year}"
- "Raccomandazioni [prodotto] economici"
- "[Prodotto] premium che vale la pena comprare"
- "Migliori offerte [prodotto] in {target_market}"

**Ricerca/Confronto (25% delle query):**
- "Migliori marche di [prodotto] in {target_market} {current_year}"
- "[Prodotto] più votati quest'anno"
- "Confrontare opzioni di [prodotto]"
- "Qual è la differenza tra [tipo A] e [tipo B]"
- "Alternative a [concorrente]"
- "Migliori marche di [prodotto] disponibili in {target_market}"

**Problema/Soluzione (20% delle query):**
- "Quale [prodotto] aiuta con [problema]"
- "Miglior [prodotto] per [esigenza specifica/condizione]"
- "Come scegliere il [prodotto] giusto"
- "Cosa cercare quando si compra [prodotto]"
- "[Problema] - quale [prodotto] raccomandano gli esperti"

**Fiducia/Recensioni (15% delle query):**
- "Marche di [prodotto] più affidabili in {target_market}"
- "I [tipo di prodotto] di {target_market} sono di buona qualità"
- "Marche di [prodotto] affidabili con buone recensioni"
- "Quali marche di [prodotto] hanno la migliore reputazione in {target_market}"
- "Marche di [prodotto] certificate in {target_market}"

AREE DI FOCUS: {focus_areas}
CONCORRENTI NOTI (usare nelle query di confronto): {competitors}

FORMATO OUTPUT:
Restituisci un array JSON dove ogni elemento ha:
- "question": il testo della query (MAI {company_name}!)
- "category": uno tra {categories}
- "intent": breve descrizione dell'obiettivo del ricercatore
- "funnel_stage": uno tra ["bottom", "middle", "top"]

IMPORTANTE: Tutte le {count} query devono essere in italiano e rilevanti per il mercato {target_market}.

Restituisci SOLO un array JSON valido, nessun altro testo.""",

    "nl": """Je bent een expert marktonderzoeker gespecialiseerd in {industry}.

CONTEXT:
- Doelmarkt/Land: {target_market}
- Geanalyseerd bedrijf: {company_name} (NOEM deze naam NIET in zoekopdrachten!)
- Bedrijfsbeschrijving: {description}
- Huidig jaar: {current_year}
- Taal: Nederlands

JOUW TAAK: Genereer {count} realistische zoekopdrachten die consumenten in {target_market} zouden stellen aan AI-assistenten (ChatGPT, Claude, Gemini, Perplexity) bij het onderzoeken of kopen van {industry} producten/diensten.

KRITIEKE REGELS:
1. NOOIT "{company_name}" opnemen in enige zoekopdracht - we testen ORGANISCHE zichtbaarheid (of AI het merk noemt zonder er specifiek naar te vragen)
2. ALLE zoekopdrachten moeten cultureel en contextueel relevant zijn voor {target_market} (lokale taalnuances, regelgeving, valuta, koopgewoonten, lokale voorkeuren)
3. Bij tijdsverwijzingen "{current_year}" of "dit jaar" gebruiken - nooit voorgaande jaren hardcoderen
4. Zoekopdrachten moeten weerspiegelen hoe echte mensen AI-assistenten vragen (conversationeel, natuurlijke taal)
5. Voeg LOKALE MARKT specifieke zoekopdrachten toe (bijv. "{target_market} regelgeving", "kopen in {target_market}", "verzending naar {target_market}", lokale winkelreferenties)

ZOEKOPDRACHT VERDELING - Genereer zoekopdrachten in deze geschatte mix:

**Koopintentie (40% van de zoekopdrachten):**
- "Beste [product] om te kopen in {target_market}"
- "Waar kan ik [product] online kopen in {target_market}"
- "[Product A] vs [Product B] - welke moet ik kopen"
- "Is [product] de prijs waard"
- "Beste prijs-kwaliteit [product] {current_year}"
- "Betaalbare [product] aanbevelingen"
- "Premium [product] de moeite waard om te kopen"
- "Beste [product] deals in {target_market}"

**Onderzoek/Vergelijking (25% van de zoekopdrachten):**
- "Beste [product] merken in {target_market} {current_year}"
- "Best beoordeelde [product] dit jaar"
- "[Product] opties vergelijken"
- "Wat is het verschil tussen [type A] en [type B]"
- "[Concurrent] alternatieven"
- "Beste [product] merken beschikbaar in {target_market}"

**Probleem/Oplossing (20% van de zoekopdrachten):**
- "Welk [product] helpt bij [probleem]"
- "Beste [product] voor [specifieke behoefte/aandoening]"
- "Hoe kies je het juiste [product]"
- "Waar op letten bij het kopen van [product]"
- "[Probleem] - welk [product] raden experts aan"

**Vertrouwen/Reviews (15% van de zoekopdrachten):**
- "Meest betrouwbare [product] merken in {target_market}"
- "Zijn [producttype] uit {target_market} van goede kwaliteit"
- "Betrouwbare [product] merken met goede reviews"
- "Welke [product] merken hebben de beste reputatie in {target_market}"
- "[Product] merken gecertificeerd in {target_market}"

FOCUSGEBIEDEN: {focus_areas}
BEKENDE CONCURRENTEN (gebruik in vergelijkingszoekopdrachten): {competitors}

UITVOERFORMAAT:
Retourneer een JSON-array waarbij elk element heeft:
- "question": de zoekopdrachttekst (NOOIT {company_name}!)
- "category": een van {categories}
- "intent": korte beschrijving van het doel van de zoeker
- "funnel_stage": een van ["bottom", "middle", "top"]

BELANGRIJK: Alle {count} zoekopdrachten moeten in het Nederlands zijn en relevant voor de {target_market} markt.

Retourneer ALLEEN een geldige JSON-array, geen andere tekst.""",

    "pl": """Jesteś ekspertem w badaniach rynku specjalizującym się w {industry}.

KONTEKST:
- Rynek docelowy/Kraj: {target_market}
- Analizowana firma: {company_name} (NIE wymieniaj tej nazwy w zapytaniach!)
- Opis działalności: {description}
- Bieżący rok: {current_year}
- Język: Polski

TWOJE ZADANIE: Wygeneruj {count} realistycznych zapytań, które konsumenci w {target_market} zadawaliby asystentom AI (ChatGPT, Claude, Gemini, Perplexity) podczas badania lub kupowania produktów/usług {industry}.

KRYTYCZNE ZASADY:
1. NIGDY nie uwzględniaj "{company_name}" w żadnym zapytaniu - testujemy ORGANICZNĄ widoczność (czy AI wspomina markę bez bezpośredniego pytania)
2. WSZYSTKIE zapytania muszą być kulturowo i kontekstowo istotne dla {target_market} (lokalne niuanse językowe, przepisy, waluta, nawyki zakupowe, lokalne preferencje)
3. Przy odniesieniach czasowych używaj "{current_year}" lub "w tym roku" - nigdy nie koduj na stałe poprzednich lat
4. Zapytania powinny odzwierciedlać sposób, w jaki prawdziwi ludzie pytają asystentów AI (konwersacyjnie, naturalny język)
5. Uwzględnij zapytania specyficzne dla RYNKU LOKALNEGO (np. "przepisy {target_market}", "kupić w {target_market}", "wysyłka do {target_market}", odniesienia do lokalnych sprzedawców)

ROZKŁAD ZAPYTAŃ - Generuj zapytania w tej przybliżonej proporcji:

**Zamiar zakupowy (40% zapytań):**
- "Najlepszy [produkt] do kupienia w {target_market}"
- "Gdzie kupić [produkt] online w {target_market}"
- "[Produkt A] vs [Produkt B] - który powinienem kupić"
- "Czy [produkt] jest wart swojej ceny"
- "Najlepsza wartość [produkt] {current_year}"
- "Przystępne cenowo rekomendacje [produkt]"
- "Premium [produkt] wart zakupu"
- "Najlepsze oferty [produkt] w {target_market}"

**Badanie/Porównanie (25% zapytań):**
- "Najlepsze marki [produkt] w {target_market} {current_year}"
- "Najwyżej oceniane [produkt] w tym roku"
- "Porównaj opcje [produkt]"
- "Jaka jest różnica między [typ A] a [typ B]"
- "Alternatywy dla [konkurent]"
- "Najlepsze marki [produkt] dostępne w {target_market}"

**Problem/Rozwiązanie (20% zapytań):**
- "Jaki [produkt] pomaga przy [problem]"
- "Najlepszy [produkt] dla [konkretna potrzeba/stan]"
- "Jak wybrać odpowiedni [produkt]"
- "Na co zwracać uwagę przy zakupie [produkt]"
- "[Problem] - jaki [produkt] polecają eksperci"

**Zaufanie/Recenzje (15% zapytań):**
- "Najbardziej zaufane marki [produkt] w {target_market}"
- "Czy [typ produktu] z {target_market} są dobrej jakości"
- "Niezawodne marki [produkt] z dobrymi recenzjami"
- "Które marki [produkt] mają najlepszą reputację w {target_market}"
- "Marki [produkt] certyfikowane w {target_market}"

OBSZARY FOKUSOWE: {focus_areas}
ZNANI KONKURENCI (użyj w zapytaniach porównawczych): {competitors}

FORMAT WYJŚCIOWY:
Zwróć tablicę JSON, gdzie każdy element ma:
- "question": tekst zapytania (NIGDY {company_name}!)
- "category": jeden z {categories}
- "intent": krótki opis celu szukającego
- "funnel_stage": jeden z ["bottom", "middle", "top"]

WAŻNE: Wszystkie {count} zapytań muszą być po polsku i istotne dla rynku {target_market}.

Zwróć TYLKO prawidłową tablicę JSON, żadnego innego tekstu.""",

    "pt": """Você é um especialista em pesquisa de mercado especializado em {industry}.

CONTEXTO:
- Mercado-alvo/País: {target_market}
- Empresa analisada: {company_name} (NÃO mencione este nome nas consultas!)
- Descrição do negócio: {description}
- Ano atual: {current_year}
- Idioma: Português

SUA TAREFA: Gere {count} consultas realistas que consumidores em {target_market} fariam a assistentes de IA (ChatGPT, Claude, Gemini, Perplexity) ao pesquisar ou comprar produtos/serviços de {industry}.

REGRAS CRÍTICAS:
1. NUNCA incluir "{company_name}" em nenhuma consulta - testamos a visibilidade ORGÂNICA (se a IA menciona a marca sem ser perguntada especificamente)
2. TODAS as consultas devem ser cultural e contextualmente relevantes para {target_market} (nuances linguísticas locais, regulamentos, moeda, hábitos de compra, preferências locais)
3. Ao fazer referências temporais, use "{current_year}" ou "este ano" - nunca codifique anos anteriores
4. As consultas devem refletir como pessoas reais perguntam a assistentes de IA (conversacional, linguagem natural)
5. Inclua consultas específicas do MERCADO LOCAL (ex: "regulamentos de {target_market}", "comprar em {target_market}", "envio para {target_market}", referências a varejistas locais)

DISTRIBUIÇÃO DE CONSULTAS - Gere consultas nesta mistura aproximada:

**Intenção de compra (40% das consultas):**
- "Melhor [produto] para comprar em {target_market}"
- "Onde comprar [produto] online em {target_market}"
- "[Produto A] vs [Produto B] - qual devo comprar"
- "[Produto] vale o preço"
- "Melhor custo-benefício [produto] {current_year}"
- "Recomendações de [produto] acessíveis"
- "[Produto] premium que vale a pena comprar"
- "Melhores ofertas de [produto] em {target_market}"

**Pesquisa/Comparação (25% das consultas):**
- "Melhores marcas de [produto] em {target_market} {current_year}"
- "[Produto] mais bem avaliados este ano"
- "Comparar opções de [produto]"
- "Qual a diferença entre [tipo A] e [tipo B]"
- "Alternativas a [concorrente]"
- "Melhores marcas de [produto] disponíveis em {target_market}"

**Problema/Solução (20% das consultas):**
- "Qual [produto] ajuda com [problema]"
- "Melhor [produto] para [necessidade específica/condição]"
- "Como escolher o [produto] certo"
- "O que procurar ao comprar [produto]"
- "[Problema] - qual [produto] os especialistas recomendam"

**Confiança/Avaliações (15% das consultas):**
- "Marcas de [produto] mais confiáveis em {target_market}"
- "[Tipo de produto] de {target_market} são de boa qualidade"
- "Marcas de [produto] confiáveis com boas avaliações"
- "Quais marcas de [produto] têm melhor reputação em {target_market}"
- "Marcas de [produto] certificadas em {target_market}"

ÁREAS DE FOCO: {focus_areas}
CONCORRENTES CONHECIDOS (use em consultas de comparação): {competitors}

FORMATO DE SAÍDA:
Retorne um array JSON onde cada elemento tem:
- "question": o texto da consulta (NUNCA {company_name}!)
- "category": um de {categories}
- "intent": breve descrição do objetivo do pesquisador
- "funnel_stage": um de ["bottom", "middle", "top"]

IMPORTANTE: Todas as {count} consultas devem estar em português e ser relevantes para o mercado {target_market}.

Retorne APENAS um array JSON válido, nenhum outro texto.""",

    "sv": """Du är en expert på marknadsundersökningar specialiserad på {industry}.

KONTEXT:
- Målmarknad/Land: {target_market}
- Analyserat företag: {company_name} (NÄMN INTE detta namn i sökfrågor!)
- Företagsbeskrivning: {description}
- Nuvarande år: {current_year}
- Språk: Svenska

DIN UPPGIFT: Generera {count} realistiska sökfrågor som konsumenter i {target_market} skulle ställa till AI-assistenter (ChatGPT, Claude, Gemini, Perplexity) när de undersöker eller köper {industry} produkter/tjänster.

KRITISKA REGLER:
1. ALDRIG inkludera "{company_name}" i någon sökfråga - vi testar ORGANISK synlighet (om AI nämner varumärket utan att bli tillfrågad specifikt)
2. ALLA sökfrågor måste vara kulturellt och kontextuellt relevanta för {target_market} (lokala språknyanser, regler, valuta, köpvanor, lokala preferenser)
3. Vid tidsreferenser, använd "{current_year}" eller "i år" - koda aldrig in tidigare år
4. Sökfrågor ska spegla hur verkliga människor frågar AI-assistenter (konversationellt, naturligt språk)
5. Inkludera LOKAL MARKNADS-specifika sökfrågor (t.ex. "{target_market} regler", "köpa i {target_market}", "leverans till {target_market}", lokala återförsäljarreferenser)

SÖKFRÅGEFÖRDELNING - Generera sökfrågor i denna ungefärliga mix:

**Köpintention (40% av sökfrågorna):**
- "Bästa [produkt] att köpa i {target_market}"
- "Var kan jag köpa [produkt] online i {target_market}"
- "[Produkt A] vs [Produkt B] - vilken ska jag köpa"
- "Är [produkt] värt priset"
- "Bästa värde [produkt] {current_year}"
- "Prisvärda [produkt] rekommendationer"
- "Premium [produkt] värt att köpa"
- "Bästa [produkt] erbjudanden i {target_market}"

**Forskning/Jämförelse (25% av sökfrågorna):**
- "Bästa [produkt] märken i {target_market} {current_year}"
- "Högst betygsatta [produkt] i år"
- "Jämför [produkt] alternativ"
- "Vad är skillnaden mellan [typ A] och [typ B]"
- "[Konkurrent] alternativ"
- "Bästa [produkt] märken tillgängliga i {target_market}"

**Problem/Lösning (20% av sökfrågorna):**
- "Vilken [produkt] hjälper mot [problem]"
- "Bästa [produkt] för [specifikt behov/tillstånd]"
- "Hur väljer man rätt [produkt]"
- "Vad ska man leta efter när man köper [produkt]"
- "[Problem] - vilken [produkt] rekommenderar experter"

**Förtroende/Recensioner (15% av sökfrågorna):**
- "Mest pålitliga [produkt] märken i {target_market}"
- "Är [produkttyp] från {target_market} av god kvalitet"
- "Pålitliga [produkt] märken med bra recensioner"
- "Vilka [produkt] märken har bäst rykte i {target_market}"
- "[Produkt] märken certifierade i {target_market}"

FOKUSOMRÅDEN: {focus_areas}
KÄNDA KONKURRENTER (använd i jämförelsefrågor): {competitors}

UTDATAFORMAT:
Returnera en JSON-array där varje element har:
- "question": sökfrågetexten (ALDRIG {company_name}!)
- "category": en av {categories}
- "intent": kort beskrivning av sökarens mål
- "funnel_stage": en av ["bottom", "middle", "top"]

VIKTIGT: Alla {count} sökfrågor måste vara på svenska och relevanta för {target_market} marknaden.

Returnera ENDAST en giltig JSON-array, ingen annan text.""",

    "da": """Du er en ekspert i markedsundersøgelser specialiseret i {industry}.

KONTEKST:
- Målmarked/Land: {target_market}
- Analyseret virksomhed: {company_name} (NÆVN IKKE dette navn i forespørgsler!)
- Virksomhedsbeskrivelse: {description}
- Nuværende år: {current_year}
- Sprog: Dansk

DIN OPGAVE: Generer {count} realistiske forespørgsler, som forbrugere i {target_market} ville stille til AI-assistenter (ChatGPT, Claude, Gemini, Perplexity) ved undersøgelse eller køb af {industry} produkter/tjenester.

KRITISKE REGLER:
1. ALDRIG inkludere "{company_name}" i nogen forespørgsel - vi tester ORGANISK synlighed (om AI nævner mærket uden at blive spurgt specifikt)
2. ALLE forespørgsler skal være kulturelt og kontekstuelt relevante for {target_market} (lokale sprognuancer, regler, valuta, købsvaner, lokale præferencer)
3. Ved tidsreferencer, brug "{current_year}" eller "i år" - aldrig hardkode tidligere år
4. Forespørgsler skal afspejle, hvordan rigtige mennesker spørger AI-assistenter (samtalebaseret, naturligt sprog)
5. Inkluder LOKALT MARKEDS-specifikke forespørgsler (f.eks. "{target_market} regler", "køb i {target_market}", "levering til {target_market}", lokale forhandlerreferencer)

FORESPØRGSELSFORDELING - Generer forespørgsler i denne omtrentlige blanding:

**Købsintention (40% af forespørgslerne):**
- "Bedste [produkt] at købe i {target_market}"
- "Hvor kan jeg købe [produkt] online i {target_market}"
- "[Produkt A] vs [Produkt B] - hvilken skal jeg købe"
- "Er [produkt] prisen værd"
- "Bedste værdi [produkt] {current_year}"
- "Overkommelige [produkt] anbefalinger"
- "Premium [produkt] værd at købe"
- "Bedste [produkt] tilbud i {target_market}"

**Forskning/Sammenligning (25% af forespørgslerne):**
- "Bedste [produkt] mærker i {target_market} {current_year}"
- "Højest bedømte [produkt] i år"
- "Sammenlign [produkt] muligheder"
- "Hvad er forskellen mellem [type A] og [type B]"
- "[Konkurrent] alternativer"
- "Bedste [produkt] mærker tilgængelige i {target_market}"

**Problem/Løsning (20% af forespørgslerne):**
- "Hvilket [produkt] hjælper med [problem]"
- "Bedste [produkt] til [specifikt behov/tilstand]"
- "Hvordan vælger man det rigtige [produkt]"
- "Hvad skal man kigge efter ved køb af [produkt]"
- "[Problem] - hvilket [produkt] anbefaler eksperter"

**Tillid/Anmeldelser (15% af forespørgslerne):**
- "Mest pålidelige [produkt] mærker i {target_market}"
- "Er [produkttype] fra {target_market} af god kvalitet"
- "Pålidelige [produkt] mærker med gode anmeldelser"
- "Hvilke [produkt] mærker har bedst omdømme i {target_market}"
- "[Produkt] mærker certificeret i {target_market}"

FOKUSOMRÅDER: {focus_areas}
KENDTE KONKURRENTER (brug i sammenligningsforespørgsler): {competitors}

OUTPUTFORMAT:
Returner et JSON-array, hvor hvert element har:
- "question": forespørgselsteksten (ALDRIG {company_name}!)
- "category": en af {categories}
- "intent": kort beskrivelse af søgerens mål
- "funnel_stage": en af ["bottom", "middle", "top"]

VIGTIGT: Alle {count} forespørgsler skal være på dansk og relevante for {target_market} markedet.

Returner KUN et gyldigt JSON-array, ingen anden tekst.""",

    "no": """Du er en ekspert på markedsundersøkelser spesialisert på {industry}.

KONTEKST:
- Målmarked/Land: {target_market}
- Analysert selskap: {company_name} (IKKE nevn dette navnet i spørringer!)
- Selskapsbeskrivelse: {description}
- Nåværende år: {current_year}
- Språk: Norsk

DIN OPPGAVE: Generer {count} realistiske spørringer som forbrukere i {target_market} ville stilt til AI-assistenter (ChatGPT, Claude, Gemini, Perplexity) når de undersøker eller kjøper {industry} produkter/tjenester.

KRITISKE REGLER:
1. ALDRI inkludere "{company_name}" i noen spørring - vi tester ORGANISK synlighet (om AI nevner merket uten å bli spurt spesifikt)
2. ALLE spørringer må være kulturelt og kontekstuelt relevante for {target_market} (lokale språknyanser, regler, valuta, kjøpsvaner, lokale preferanser)
3. Ved tidsreferanser, bruk "{current_year}" eller "i år" - aldri hardkode tidligere år
4. Spørringer skal gjenspeile hvordan ekte mennesker spør AI-assistenter (samtalebasert, naturlig språk)
5. Inkluder LOKALT MARKEDS-spesifikke spørringer (f.eks. "{target_market} regler", "kjøpe i {target_market}", "levering til {target_market}", lokale forhandlerreferanser)

SPØRRINGSFORDELING - Generer spørringer i denne omtrentlige blandingen:

**Kjøpsintensjon (40% av spørringene):**
- "Beste [produkt] å kjøpe i {target_market}"
- "Hvor kan jeg kjøpe [produkt] online i {target_market}"
- "[Produkt A] vs [Produkt B] - hvilken bør jeg kjøpe"
- "Er [produkt] verdt prisen"
- "Beste verdi [produkt] {current_year}"
- "Rimelige [produkt] anbefalinger"
- "Premium [produkt] verdt å kjøpe"
- "Beste [produkt] tilbud i {target_market}"

**Forskning/Sammenligning (25% av spørringene):**
- "Beste [produkt] merker i {target_market} {current_year}"
- "Høyest rangerte [produkt] i år"
- "Sammenlign [produkt] alternativer"
- "Hva er forskjellen mellom [type A] og [type B]"
- "[Konkurrent] alternativer"
- "Beste [produkt] merker tilgjengelig i {target_market}"

**Problem/Løsning (20% av spørringene):**
- "Hvilket [produkt] hjelper med [problem]"
- "Beste [produkt] for [spesifikt behov/tilstand]"
- "Hvordan velge riktig [produkt]"
- "Hva skal man se etter når man kjøper [produkt]"
- "[Problem] - hvilket [produkt] anbefaler eksperter"

**Tillit/Anmeldelser (15% av spørringene):**
- "Mest pålitelige [produkt] merker i {target_market}"
- "Er [produkttype] fra {target_market} av god kvalitet"
- "Pålitelige [produkt] merker med gode anmeldelser"
- "Hvilke [produkt] merker har best omdømme i {target_market}"
- "[Produkt] merker sertifisert i {target_market}"

FOKUSOMRÅDER: {focus_areas}
KJENTE KONKURRENTER (bruk i sammenligningsspørringer): {competitors}

OUTPUTFORMAT:
Returner et JSON-array der hvert element har:
- "question": spørringsteksten (ALDRI {company_name}!)
- "category": en av {categories}
- "intent": kort beskrivelse av søkerens mål
- "funnel_stage": en av ["bottom", "middle", "top"]

VIKTIG: Alle {count} spørringer må være på norsk og relevante for {target_market} markedet.

Returner KUN et gyldig JSON-array, ingen annen tekst.""",

    "fi": """Olet markkinatutkimuksen asiantuntija, joka on erikoistunut alaan {industry}.

KONTEKSTI:
- Kohdemarkkina/Maa: {target_market}
- Analysoitava yritys: {company_name} (ÄLÄ mainitse tätä nimeä kyselyissä!)
- Yrityksen kuvaus: {description}
- Kuluva vuosi: {current_year}
- Kieli: Suomi

TEHTÄVÄSI: Luo {count} realistista kyselyä, joita kuluttajat {target_market}:ssa esittäisivät tekoälyavustajille (ChatGPT, Claude, Gemini, Perplexity) tutkiessaan tai ostaessaan {industry} tuotteita/palveluita.

KRIITTISET SÄÄNNÖT:
1. ÄLÄ KOSKAAN sisällytä "{company_name}" mihinkään kyselyyn - testaamme ORGAANISTA näkyvyyttä (mainitseeko tekoäly brändin ilman, että sitä kysytään erikseen)
2. KAIKKIEN kyselyjen on oltava kulttuurisesti ja kontekstuaalisesti relevantteja {target_market}:lle (paikalliset kielten vivahteet, säädökset, valuutta, ostotavat, paikalliset mieltymykset)
3. Aikaviittauksissa käytä "{current_year}" tai "tänä vuonna" - älä koskaan kovakoodaa menneitä vuosia
4. Kyselyjen tulee heijastaa, miten oikeat ihmiset kysyvät tekoälyavustajilta (keskustelullinen, luonnollinen kieli)
5. Sisällytä PAIKALLISMARKKINOILLE ominaisia kyselyitä (esim. "{target_market} säädökset", "ostaa {target_market}:sta", "toimitus {target_market}:iin", paikalliset jälleenmyyjäviittaukset)

KYSELYJEN JAKAUMA - Luo kyselyjä tässä likimääräisessä sekoituksessa:

**Ostoaikomus (40% kyselyistä):**
- "Paras [tuote] ostettavaksi {target_market}:sta"
- "Mistä voin ostaa [tuote] verkosta {target_market}:ssa"
- "[Tuote A] vs [Tuote B] - kumman minun pitäisi ostaa"
- "Onko [tuote] hintansa arvoinen"
- "Paras hinta-laatusuhde [tuote] {current_year}"
- "Edulliset [tuote] suositukset"
- "Premium [tuote] ostamisen arvoinen"
- "Parhaat [tuote] tarjoukset {target_market}:ssa"

**Tutkimus/Vertailu (25% kyselyistä):**
- "Parhaat [tuote] merkit {target_market}:ssa {current_year}"
- "Parhaiten arvioidut [tuote] tänä vuonna"
- "Vertaa [tuote] vaihtoehtoja"
- "Mikä on ero [tyyppi A]:n ja [tyyppi B]:n välillä"
- "[Kilpailija] vaihtoehdot"
- "Parhaat [tuote] merkit saatavilla {target_market}:ssa"

**Ongelma/Ratkaisu (20% kyselyistä):**
- "Mikä [tuote] auttaa [ongelma]:an"
- "Paras [tuote] [erityistarve/tila]:lle"
- "Kuinka valita oikea [tuote]"
- "Mitä etsiä ostaessa [tuote]"
- "[Ongelma] - mitä [tuote] asiantuntijat suosittelevat"

**Luottamus/Arvostelut (15% kyselyistä):**
- "Luotettavimmat [tuote] merkit {target_market}:ssa"
- "Ovatko [tuotetyyppi] {target_market}:sta laadukkaita"
- "Luotettavat [tuote] merkit hyvillä arvosteluilla"
- "Millä [tuote] merkeillä on paras maine {target_market}:ssa"
- "[Tuote] merkit sertifioitu {target_market}:ssa"

PAINOPISTEALUEET: {focus_areas}
TUNNETUT KILPAILIJAT (käytä vertailukyselyissä): {competitors}

TULOSMUOTO:
Palauta JSON-taulukko, jossa jokaisella elementillä on:
- "question": kyselyteksti (EI KOSKAAN {company_name}!)
- "category": yksi seuraavista {categories}
- "intent": lyhyt kuvaus hakijan tavoitteesta
- "funnel_stage": yksi seuraavista ["bottom", "middle", "top"]

TÄRKEÄÄ: Kaikkien {count} kyselyn on oltava suomeksi ja relevantteja {target_market} markkinoille.

Palauta VAIN kelvollinen JSON-taulukko, ei muuta tekstiä.""",

    "ja": """あなたは{industry}を専門とする市場調査のエキスパートです。

コンテキスト:
- ターゲット市場/国: {target_market}
- 分析対象企業: {company_name}（このブランド名をクエリに含めないでください！）
- 事業内容: {description}
- 現在の年: {current_year}
- 言語: 日本語

あなたのタスク: {target_market}の消費者が{industry}の製品/サービスを調査または購入する際に、AIアシスタント（ChatGPT、Claude、Gemini、Perplexity）に尋ねる現実的なクエリを{count}個生成してください。

重要なルール:
1. いかなるクエリにも「{company_name}」を含めないでください - オーガニックな可視性をテストしています（AIが特に尋ねられなくてもブランドを言及するかどうか）
2. すべてのクエリは{target_market}に文化的・文脈的に関連している必要があります（現地の言語ニュアンス、規制、通貨、購買習慣、現地の嗜好）
3. 時間を参照する際は「{current_year}」または「今年」を使用してください - 過去の年をハードコードしないでください
4. クエリは実際の人々がAIアシスタントに質問する方法を反映する必要があります（会話的、自然な言語）
5. ローカル市場特有のクエリを含めてください（例：「{target_market}の規制」、「{target_market}で購入」、「{target_market}への配送」、現地小売業者への言及）

クエリの分布 - この概算の組み合わせでクエリを生成してください:

**購買意図（クエリの40%）:**
- 「{target_market}で買うべき最高の[製品]」
- 「{target_market}でオンラインで[製品]を購入できる場所」
- 「[製品A] vs [製品B] - どちらを買うべき」
- 「[製品]は価格に見合う価値があるか」
- 「{current_year}のベストバリュー[製品]」
- 「手頃な価格の[製品]のおすすめ」
- 「購入する価値のあるプレミアム[製品]」
- 「{target_market}での最高の[製品]セール」

**調査/比較（クエリの25%）:**
- 「{target_market}の{current_year}ベスト[製品]ブランド」
- 「今年最も評価の高い[製品]」
- 「[製品]オプションを比較」
- 「[タイプA]と[タイプB]の違いは何ですか」
- 「[競合他社]の代替品」
- 「{target_market}で入手可能な最高の[製品]ブランド」

**問題/解決（クエリの20%）:**
- 「[問題]に役立つ[製品]は何ですか」
- 「[特定のニーズ/状態]に最適な[製品]」
- 「適切な[製品]の選び方」
- 「[製品]を購入する際に何を探すべきか」
- 「[問題] - 専門家が推奨する[製品]は何ですか」

**信頼/レビュー（クエリの15%）:**
- 「{target_market}で最も信頼できる[製品]ブランド」
- 「{target_market}の[製品タイプ]は品質が良いですか」
- 「良いレビューのある信頼できる[製品]ブランド」
- 「{target_market}で最高の評判を持つ[製品]ブランドはどれですか」
- 「{target_market}で認定された[製品]ブランド」

フォーカスエリア: {focus_areas}
既知の競合他社（比較クエリで使用）: {competitors}

出力形式:
各要素が以下を持つJSON配列を返してください:
- "question": クエリテキスト（絶対に{company_name}を含めないでください！）
- "category": {categories}のいずれか
- "intent": 検索者の目標の簡単な説明
- "funnel_stage": ["bottom", "middle", "top"]のいずれか

重要: すべての{count}クエリは日本語で、{target_market}市場に関連している必要があります。

有効なJSON配列のみを返してください。他のテキストは含めないでください。""",

    "zh": """您是专门研究{industry}的市场研究专家。

背景:
- 目标市场/国家: {target_market}
- 被分析的公司: {company_name}（不要在查询中提及此名称！）
- 业务描述: {description}
- 当前年份: {current_year}
- 语言: 中文

您的任务: 生成{count}个真实的查询，这些查询是{target_market}的消费者在研究或购买{industry}产品/服务时会向AI助手（ChatGPT、Claude、Gemini、Perplexity）提出的。

关键规则:
1. 任何查询中都不要包含"{company_name}" - 我们正在测试有机可见性（AI是否在未被特别询问的情况下提及品牌）
2. 所有查询必须与{target_market}在文化和语境上相关（当地语言细微差别、法规、货币、购买习惯、当地偏好）
3. 在引用时间时，使用"{current_year}"或"今年" - 永远不要硬编码过去的年份
4. 查询应反映真实的人如何询问AI助手（对话式、自然语言）
5. 包括本地市场特定的查询（例如："{target_market}法规"、"在{target_market}购买"、"运送到{target_market}"、当地零售商参考）

查询分布 - 按此大致比例生成查询:

**购买意图（40%的查询）:**
- "在{target_market}购买最好的[产品]"
- "在{target_market}哪里可以在线购买[产品]"
- "[产品A] vs [产品B] - 我应该买哪个"
- "[产品]值这个价吗"
- "{current_year}最佳性价比[产品]"
- "实惠的[产品]推荐"
- "值得购买的高端[产品]"
- "{target_market}最佳[产品]优惠"

**研究/比较（25%的查询）:**
- "{target_market} {current_year}最佳[产品]品牌"
- "今年评分最高的[产品]"
- "比较[产品]选项"
- "[类型A]和[类型B]有什么区别"
- "[竞争对手]的替代品"
- "{target_market}可用的最佳[产品]品牌"

**问题/解决方案（20%的查询）:**
- "什么[产品]有助于[问题]"
- "最适合[特定需求/状况]的[产品]"
- "如何选择合适的[产品]"
- "购买[产品]时要注意什么"
- "[问题] - 专家推荐什么[产品]"

**信任/评论（15%的查询）:**
- "{target_market}最值得信赖的[产品]品牌"
- "{target_market}的[产品类型]质量好吗"
- "有良好评价的可靠[产品]品牌"
- "哪些[产品]品牌在{target_market}声誉最好"
- "在{target_market}获得认证的[产品]品牌"

重点领域: {focus_areas}
已知竞争对手（在比较查询中使用）: {competitors}

输出格式:
返回一个JSON数组，其中每个元素包含:
- "question": 查询文本（永远不要包含{company_name}！）
- "category": {categories}之一
- "intent": 搜索者目标的简要描述
- "funnel_stage": ["bottom", "middle", "top"]之一

重要: 所有{count}个查询必须是中文，并且与{target_market}市场相关。

只返回有效的JSON数组，不要包含其他文本。""",

    "ko": """당신은 {industry}을(를) 전문으로 하는 시장 조사 전문가입니다.

컨텍스트:
- 목표 시장/국가: {target_market}
- 분석 대상 회사: {company_name} (이 이름을 쿼리에 포함하지 마세요!)
- 사업 설명: {description}
- 현재 연도: {current_year}
- 언어: 한국어

당신의 과제: {target_market}의 소비자들이 {industry} 제품/서비스를 조사하거나 구매할 때 AI 어시스턴트(ChatGPT, Claude, Gemini, Perplexity)에게 물을 수 있는 현실적인 쿼리를 {count}개 생성하세요.

중요한 규칙:
1. 어떤 쿼리에도 "{company_name}"을(를) 포함하지 마세요 - 우리는 유기적 가시성을 테스트하고 있습니다 (AI가 특별히 질문받지 않고도 브랜드를 언급하는지)
2. 모든 쿼리는 {target_market}에 문화적, 맥락적으로 관련이 있어야 합니다 (현지 언어 뉘앙스, 규정, 통화, 구매 습관, 현지 선호도)
3. 시간을 언급할 때 "{current_year}" 또는 "올해"를 사용하세요 - 과거 연도를 하드코딩하지 마세요
4. 쿼리는 실제 사람들이 AI 어시스턴트에게 질문하는 방식을 반영해야 합니다 (대화체, 자연스러운 언어)
5. 지역 시장 특정 쿼리를 포함하세요 (예: "{target_market} 규정", "{target_market}에서 구매", "{target_market}으로 배송", 현지 소매업체 참조)

쿼리 분포 - 이 대략적인 비율로 쿼리를 생성하세요:

**구매 의도 (쿼리의 40%):**
- "{target_market}에서 구매할 최고의 [제품]"
- "{target_market}에서 온라인으로 [제품]을 어디서 살 수 있나요"
- "[제품 A] vs [제품 B] - 어떤 것을 사야 하나요"
- "[제품]이 가격만큼 가치가 있나요"
- "{current_year} 최고의 가성비 [제품]"
- "합리적인 가격의 [제품] 추천"
- "구매할 가치가 있는 프리미엄 [제품]"
- "{target_market}에서 최고의 [제품] 거래"

**조사/비교 (쿼리의 25%):**
- "{target_market} {current_year} 최고의 [제품] 브랜드"
- "올해 가장 높은 평점을 받은 [제품]"
- "[제품] 옵션 비교"
- "[유형 A]와 [유형 B]의 차이점은 무엇인가요"
- "[경쟁사] 대안"
- "{target_market}에서 구할 수 있는 최고의 [제품] 브랜드"

**문제/해결 (쿼리의 20%):**
- "어떤 [제품]이 [문제]에 도움이 되나요"
- "[특정 필요/상태]에 가장 좋은 [제품]"
- "올바른 [제품]을 선택하는 방법"
- "[제품]을 구매할 때 무엇을 찾아야 하나요"
- "[문제] - 전문가들이 추천하는 [제품]은 무엇인가요"

**신뢰/리뷰 (쿼리의 15%):**
- "{target_market}에서 가장 신뢰할 수 있는 [제품] 브랜드"
- "{target_market}의 [제품 유형]은 품질이 좋은가요"
- "좋은 리뷰를 가진 신뢰할 수 있는 [제품] 브랜드"
- "{target_market}에서 가장 좋은 평판을 가진 [제품] 브랜드는 무엇인가요"
- "{target_market}에서 인증된 [제품] 브랜드"

집중 영역: {focus_areas}
알려진 경쟁사 (비교 쿼리에 사용): {competitors}

출력 형식:
각 요소가 다음을 포함하는 JSON 배열을 반환하세요:
- "question": 쿼리 텍스트 (절대로 {company_name}을(를) 포함하지 마세요!)
- "category": {categories} 중 하나
- "intent": 검색자 목표에 대한 간략한 설명
- "funnel_stage": ["bottom", "middle", "top"] 중 하나

중요: 모든 {count}개의 쿼리는 한국어여야 하고 {target_market} 시장과 관련이 있어야 합니다.

유효한 JSON 배열만 반환하세요. 다른 텍스트는 포함하지 마세요.""",

    "ar": """أنت خبير في أبحاث السوق متخصص في {industry}.

السياق:
- السوق المستهدف/البلد: {target_market}
- الشركة التي يتم تحليلها: {company_name} (لا تذكر هذا الاسم في الاستعلامات!)
- وصف العمل: {description}
- السنة الحالية: {current_year}
- اللغة: العربية

مهمتك: قم بإنشاء {count} استعلامات واقعية يطرحها المستهلكون في {target_market} على مساعدي الذكاء الاصطناعي (ChatGPT، Claude، Gemini، Perplexity) عند البحث عن أو شراء منتجات/خدمات {industry}.

القواعد الحاسمة:
1. لا تضمن "{company_name}" في أي استعلام أبداً - نحن نختبر الظهور العضوي (هل يذكر الذكاء الاصطناعي العلامة التجارية دون أن يُسأل عنها تحديداً)
2. يجب أن تكون جميع الاستعلامات ذات صلة ثقافية وسياقية بـ {target_market} (الفروق اللغوية المحلية، اللوائح، العملة، عادات الشراء، التفضيلات المحلية)
3. عند الإشارة إلى الوقت، استخدم "{current_year}" أو "هذا العام" - لا تضع سنوات سابقة بشكل ثابت أبداً
4. يجب أن تعكس الاستعلامات كيف يسأل الناس الحقيقيون مساعدي الذكاء الاصطناعي (حوارية، لغة طبيعية)
5. قم بتضمين استعلامات خاصة بالسوق المحلي (مثل: "لوائح {target_market}"، "الشراء في {target_market}"، "الشحن إلى {target_market}"، مراجع تجار التجزئة المحليين)

توزيع الاستعلامات - قم بإنشاء استعلامات بهذا المزيج التقريبي:

**نية الشراء (40% من الاستعلامات):**
- "أفضل [منتج] للشراء في {target_market}"
- "أين يمكنني شراء [منتج] عبر الإنترنت في {target_market}"
- "[منتج أ] مقابل [منتج ب] - أيهما يجب أن أشتري"
- "هل [منتج] يستحق السعر"
- "أفضل قيمة مقابل المال [منتج] {current_year}"
- "توصيات [منتج] بأسعار معقولة"
- "[منتج] فاخر يستحق الشراء"
- "أفضل عروض [منتج] في {target_market}"

**البحث/المقارنة (25% من الاستعلامات):**
- "أفضل علامات [منتج] التجارية في {target_market} {current_year}"
- "أعلى تقييم [منتج] هذا العام"
- "مقارنة خيارات [منتج]"
- "ما الفرق بين [نوع أ] و [نوع ب]"
- "بدائل [منافس]"
- "أفضل علامات [منتج] التجارية المتاحة في {target_market}"

**المشكلة/الحل (20% من الاستعلامات):**
- "ما [منتج] يساعد في [مشكلة]"
- "أفضل [منتج] لـ [حاجة محددة/حالة]"
- "كيف تختار [منتج] المناسب"
- "ما الذي يجب البحث عنه عند شراء [منتج]"
- "[مشكلة] - ما [منتج] يوصي به الخبراء"

**الثقة/المراجعات (15% من الاستعلامات):**
- "أكثر علامات [منتج] التجارية موثوقية في {target_market}"
- "هل [نوع المنتج] من {target_market} ذو جودة عالية"
- "علامات [منتج] التجارية الموثوقة مع مراجعات جيدة"
- "أي علامات [منتج] التجارية لديها أفضل سمعة في {target_market}"
- "علامات [منتج] التجارية المعتمدة في {target_market}"

مجالات التركيز: {focus_areas}
المنافسون المعروفون (استخدم في استعلامات المقارنة): {competitors}

تنسيق الإخراج:
أعد مصفوفة JSON حيث يحتوي كل عنصر على:
- "question": نص الاستعلام (لا تضمن {company_name} أبداً!)
- "category": واحدة من {categories}
- "intent": وصف موجز لهدف الباحث
- "funnel_stage": واحدة من ["bottom", "middle", "top"]

مهم: يجب أن تكون جميع الاستعلامات الـ {count} باللغة العربية وذات صلة بسوق {target_market}.

أعد مصفوفة JSON صالحة فقط، بدون أي نص آخر.""",

    "hi": """आप {industry} में विशेषज्ञता रखने वाले मार्केट रिसर्च एक्सपर्ट हैं।

संदर्भ:
- लक्ष्य बाजार/देश: {target_market}
- विश्लेषित कंपनी: {company_name} (इस नाम को क्वेरी में शामिल न करें!)
- व्यवसाय विवरण: {description}
- वर्तमान वर्ष: {current_year}
- भाषा: हिंदी

आपका कार्य: {count} यथार्थवादी क्वेरी जनरेट करें जो {target_market} के उपभोक्ता {industry} उत्पादों/सेवाओं की खोज या खरीद करते समय AI असिस्टेंट (ChatGPT, Claude, Gemini, Perplexity) से पूछेंगे।

महत्वपूर्ण नियम:
1. किसी भी क्वेरी में "{company_name}" शामिल न करें - हम ऑर्गेनिक विजिबिलिटी का परीक्षण कर रहे हैं (क्या AI बिना पूछे ब्रांड का उल्लेख करता है)
2. सभी क्वेरी {target_market} के लिए सांस्कृतिक और संदर्भात्मक रूप से प्रासंगिक होनी चाहिए (स्थानीय भाषा की बारीकियां, नियम, मुद्रा, खरीदारी की आदतें, स्थानीय प्राथमिकताएं)
3. समय का संदर्भ देते समय "{current_year}" या "इस साल" का उपयोग करें - पिछले वर्षों को हार्डकोड न करें
4. क्वेरी को दर्शाना चाहिए कि वास्तविक लोग AI असिस्टेंट से कैसे पूछते हैं (बातचीत की शैली, प्राकृतिक भाषा)
5. स्थानीय बाजार विशिष्ट क्वेरी शामिल करें (जैसे: "{target_market} नियम", "{target_market} में खरीदें", "{target_market} में डिलीवरी", स्थानीय रिटेलर संदर्भ)

क्वेरी वितरण - इस अनुमानित मिश्रण में क्वेरी जनरेट करें:

**खरीद इरादा (40% क्वेरी):**
- "{target_market} में खरीदने के लिए सबसे अच्छा [उत्पाद]"
- "{target_market} में ऑनलाइन [उत्पाद] कहां से खरीदें"
- "[उत्पाद A] vs [उत्पाद B] - कौन सा खरीदना चाहिए"
- "क्या [उत्पाद] कीमत के लायक है"
- "{current_year} में सबसे अच्छी वैल्यू [उत्पाद]"
- "किफायती [उत्पाद] सिफारिशें"
- "प्रीमियम [उत्पाद] खरीदने लायक"
- "{target_market} में सबसे अच्छे [उत्पाद] डील्स"

**रिसर्च/तुलना (25% क्वेरी):**
- "{target_market} में {current_year} के सबसे अच्छे [उत्पाद] ब्रांड"
- "इस साल सबसे ज्यादा रेटेड [उत्पाद]"
- "[उत्पाद] विकल्पों की तुलना करें"
- "[टाइप A] और [टाइप B] में क्या अंतर है"
- "[प्रतिस्पर्धी] के विकल्प"
- "{target_market} में उपलब्ध सबसे अच्छे [उत्पाद] ब्रांड"

**समस्या/समाधान (20% क्वेरी):**
- "कौन सा [उत्पाद] [समस्या] में मदद करता है"
- "[विशिष्ट जरूरत/स्थिति] के लिए सबसे अच्छा [उत्पाद]"
- "सही [उत्पाद] कैसे चुनें"
- "[उत्पाद] खरीदते समय क्या देखें"
- "[समस्या] - विशेषज्ञ कौन सा [उत्पाद] सुझाते हैं"

**विश्वास/समीक्षा (15% क्वेरी):**
- "{target_market} में सबसे भरोसेमंद [उत्पाद] ब्रांड"
- "क्या {target_market} के [उत्पाद प्रकार] अच्छी गुणवत्ता के हैं"
- "अच्छी समीक्षाओं वाले विश्वसनीय [उत्पाद] ब्रांड"
- "{target_market} में कौन से [उत्पाद] ब्रांड की सबसे अच्छी प्रतिष्ठा है"
- "{target_market} में प्रमाणित [उत्पाद] ब्रांड"

फोकस क्षेत्र: {focus_areas}
ज्ञात प्रतिस्पर्धी (तुलना क्वेरी में उपयोग करें): {competitors}

आउटपुट फॉर्मेट:
एक JSON ऐरे लौटाएं जहां प्रत्येक तत्व में हो:
- "question": क्वेरी टेक्स्ट (कभी भी {company_name} शामिल न करें!)
- "category": {categories} में से एक
- "intent": खोजकर्ता के लक्ष्य का संक्षिप्त विवरण
- "funnel_stage": ["bottom", "middle", "top"] में से एक

महत्वपूर्ण: सभी {count} क्वेरी हिंदी में होनी चाहिए और {target_market} बाजार के लिए प्रासंगिक होनी चाहिए।

केवल वैध JSON ऐरे लौटाएं, कोई अन्य टेक्स्ट नहीं।""",
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


def _get_current_year() -> str:
    """Get the current year as a string."""
    return str(datetime.now().year)


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
        List of query dictionaries with question, category, intent, funnel_stage
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

    # Get current year
    current_year = _get_current_year()

    # Build prompt
    prompt = template.format(
        industry=context.industry,
        company_name=context.company_name,
        description=context.description or f"A company in the {context.industry} industry",
        target_market=context.target_market or "General consumers",
        focus_areas=", ".join(context.focus_areas) if context.focus_areas else "All product areas",
        competitors=", ".join(context.competitor_names) if context.competitor_names else "Unknown",
        count=count,
        categories=", ".join(QUERY_CATEGORIES),
        current_year=current_year
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
                q["category"] = "shopping_intent"
            if "funnel_stage" not in q:
                q["funnel_stage"] = "middle"

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
        List of query dictionaries with question, category, intent, funnel_stage
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

    # Get current year
    current_year = _get_current_year()

    # Build prompt
    prompt = template.format(
        industry=context.industry,
        company_name=context.company_name,
        description=context.description or f"A company in the {context.industry} industry",
        target_market=context.target_market or "General consumers",
        focus_areas=", ".join(context.focus_areas) if context.focus_areas else "All product areas",
        competitors=", ".join(context.competitor_names) if context.competitor_names else "Unknown",
        count=count,
        categories=", ".join(QUERY_CATEGORIES),
        current_year=current_year
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
                q["category"] = "shopping_intent"
            if "funnel_stage" not in q:
                q["funnel_stage"] = "middle"

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
    industry = context.industry
    current_year = _get_current_year()
    target_market = context.target_market or "your region"

    # Language-specific query templates with dynamic year
    lang = context.language.lower()[:2] if context.language else "en"

    if lang == "de":
        templates = [
            f"Was sind die besten {industry} Marken in {target_market}?",
            f"Vergleich {industry} Produkte {current_year}",
            f"Wo kann ich {industry} Produkte online in {target_market} kaufen?",
            f"Worauf sollte ich beim Kauf von {industry} achten?",
            f"Beste {industry} für Anfänger {current_year}",
            f"Top bewertete {industry} Marken dieses Jahr",
            f"Natürliche {industry} Empfehlungen",
            f"Bio {industry} Marken {target_market}",
            f"{industry} Qualitätsvergleich {current_year}",
            f"Beste Preis-Leistung {industry}",
            f"Premium {industry} Marken Test {current_year}",
            f"Welche {industry} Marke ist am vertrauenswürdigsten?",
            f"Kundenbewertungen {industry} {current_year}",
            f"Beste {industry} für Gesundheit",
            f"Nachhaltige {industry} Produkte {target_market}",
            f"Vegane {industry} Alternativen",
            f"Günstige {industry} Empfehlungen {current_year}",
            f"{industry} ohne Zusatzstoffe",
            f"Hochwertige {industry} kaufen in {target_market}",
            f"Empfehlenswerte {industry} Anbieter {current_year}",
        ]
    else:
        templates = [
            f"What are the best {industry} brands in {target_market}?",
            f"Compare {industry} products {current_year}",
            f"Where can I buy {industry} products online in {target_market}?",
            f"What should I look for when buying {industry}?",
            f"Best {industry} for beginners {current_year}",
            f"Top rated {industry} brands this year",
            f"Natural {industry} recommendations",
            f"Organic {industry} brands in {target_market}",
            f"{industry} quality comparison {current_year}",
            f"Best value {industry} products",
            f"Premium {industry} brands review {current_year}",
            f"Which {industry} brand is most trustworthy?",
            f"Customer reviews for {industry} {current_year}",
            f"Best {industry} for health benefits",
            f"Sustainable {industry} products in {target_market}",
            f"Vegan {industry} alternatives",
            f"Budget-friendly {industry} recommendations {current_year}",
            f"{industry} without additives",
            f"High-quality {industry} to buy in {target_market}",
            f"Recommended {industry} providers {current_year}",
        ]

    queries = []
    funnel_stages = ["bottom", "bottom", "middle", "middle", "top"]

    for i, q in enumerate(templates[:count]):
        category = QUERY_CATEGORIES[i % len(QUERY_CATEGORIES)]
        funnel_stage = funnel_stages[i % len(funnel_stages)]
        queries.append({
            "question": q,
            "category": category,
            "prompt_id": f"sample_{i+1}",
            "intent": f"User researching {industry}",
            "funnel_stage": funnel_stage,
            "generated_by": "fallback"
        })

    return queries
