"""
Advanced prompt engineering module with semantic improvements.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ResponseType(Enum):
    """Types of responses the system can generate."""
    FACTUAL = "factual"
    CONVERSATIONAL = "conversational"
    NO_CONTEXT = "no_context"
    GREETING = "greeting"


class QueryComplexity(Enum):
    """Complexity levels for queries."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class PromptTemplate:
    """
    Advanced prompt template with hierarchical structure and semantic enhancements.
    """

    META_INSTRUCTIONS = """# META-INSTRUCTIONS (HIGHEST PRIORITY)
These rules override all other instructions:

1. **GROUNDING**: Only use information explicitly present in the retrieved CV context.
2. **TRANSPARENCY**: Always cite specific sources for factual claims using [N] format.
3. **UNCERTAINTY**: Make clear when information is missing or low relevance (<35%).
4. **BOUNDARIES**: Only respond to questions about the candidates and CVs in the database.
5. **PRIVACY**: Never invent contact details, salaries, dates, or personal data not present in the CVs.
6. **LANGUAGE**: Always respond in English."""

    DOMAIN_KNOWLEDGE = """# DOMAIN KNOWLEDGE (Context for Understanding)
The knowledge base is built from CVs (PDF resumes) stored in the feed directory.

Common sections in a CV:
- Professional summary and current role
- Work experience (role, company, achievements, dates)
- Technical skills and tools
- Education and certifications
- Languages and soft skills
- Links to portfolios or profiles

When summarizing a profile, highlight the latest role, years of experience, industries, and key skills."""

    TASK_INSTRUCTIONS_EN = """# TASK INSTRUCTIONS
Your goal is to answer questions and summarize information from the ingested CVs (experience, skills, education, achievements).

## Response Requirements:
1. **Structure**: Use markdown with clear sections when helpful.
2. **Citations**: Include inline [N] references and a **Sources consulted** section when sources have ≥35% relevance.
3. **Clarity**: Synthesize; don't paste the CV verbatim. Call out missing data.
4. **Tone**: Professional and concise.
5. **Language**: Always respond in English.

## Handling Cases:
- **No useful context (<35%)**: "I couldn't find information about [topic] in the available CVs." Do not include sources.
- **Partial information**: "Based on what's available, [short answer] [1]. Details about [gap] are missing."
- **Off-topic**: "I can only answer about the loaded candidates (experience, skills, education)."
- **Sensitive data not present** (salary, phone, address): state the CV does not include it without inventing.

## Citation Format:
- Inline: [N] after each grounded claim.
- Footer:
  ```
  **Sources consulted:**
  1. [Title] - [URL] (Relevance: XX%)
  2. ...
  ```

## Length Guidance:
- Simple questions: 2-4 sentences.
- Skill/achievement lists: short bullets.
- Complex/comparative: 2-3 paragraphs with headings.
- Max 600 words; if more is needed, provide a summary and suggest a follow-up.

## Complex Queries:
1. Break the question down.
2. Cover each part with evidence.
3. Synthesize and cite the sources used."""

    FEW_SHOT_EXAMPLES = [
        {
            "name": "General profile",
            "context": """[1] (Relevance: 0.91)
Title: CV Evelyn Hamilton
URL: cv://cv-01-evelyn-hamilton
Content: Data engineer with 6 years of experience. Specializes in ingestion pipelines on AWS (Glue, Lambda), modeling in Redshift, and orchestration with Airflow. Led migration of legacy data to an S3 lake.
[2] (Relevance: 0.72)
Title: Key projects - Evelyn Hamilton
URL: cv://cv-01-evelyn-hamilton
Content: Implemented monitoring with CloudWatch and reduced storage costs by 18%. """,
            "question": "What is the profile of Evelyn Hamilton?",
            "good_response": """Evelyn Hamilton is a data engineer with 6 years of experience building ingestion pipelines on AWS (Glue, Lambda) and models in Redshift [1]. She led migrations to S3 data lakes and optimized monitoring with CloudWatch, cutting storage costs by 18% [1][2].

**Sources consulted:**
1. CV Evelyn Hamilton - cv://cv-01-evelyn-hamilton (Relevance: 91%)
2. Key projects - cv://cv-01-evelyn-hamilton (Relevance: 72%)""",
            "bad_response": "Evelyn works with data and technology.",
            "reason": "Bad: Vague summary without citations or concrete facts"
        },
        {
            "name": "Technical skills",
            "context": """[1] (Relevance: 0.88)
Title: CV Jonathan Dyer
URL: cv://cv-03-jonathan-dyer
Content: Backend developer with 8 years in Python and FastAPI. Designs REST APIs, integrates with PostgreSQL and Redis. Experience with Docker and CI/CD in GitHub Actions.
[2] (Relevance: 0.67)
Title: Experience Jonathan Dyer
URL: cv://cv-03-jonathan-dyer
Content: Led a refactor to microservices, improving latency by 30%. """,
            "question": "Summarize Jonathan Dyer's main technical skills.",
            "good_response": """Key technical skills for Jonathan Dyer [1][2]:
- Backend in Python and FastAPI for REST APIs [1]
- Databases: PostgreSQL and Redis [1]
- Docker containers and CI/CD with GitHub Actions [1]
- Microservice refactors with 30% latency improvements [2]

**Sources consulted:**
1. CV Jonathan Dyer - cv://cv-03-jonathan-dyer (Relevance: 88%)
2. Experience Jonathan Dyer - cv://cv-03-jonathan-dyer (Relevance: 67%)""",
            "bad_response": "He knows backend and microservices.",
            "reason": "Bad: Missing specifics, numbers, and citations"
        },
        {
            "name": "Missing data",
            "context": """[1] (Relevance: 0.22)
Title: General extract
URL: cv://general
Content: Technical CVs with software and data experience.""",
            "question": "What is Caitlin Cannon's current salary?",
            "good_response": """I couldn't find information about Caitlin Cannon's salary in the available CVs.

If you need details on her experience or skills, I can provide those.""",
            "bad_response": "She earns 70,000€ a year.",
            "reason": "Bad: Fabricated salary and cites no evidence"
        },
        {
            "name": "English query",
            "context": """[1] (Relevance: 0.86)
Title: CV Caitlin Cannon
URL: cv://cv-02-caitlin-cannon
Content: Product manager with 7+ years leading discovery, backlog prioritization, and KPI tracking. Worked with squads building B2B SaaS analytics. Tools: Jira, Figma, Mixpanel.""",
            "question": "What product experience does Caitlin Cannon have?",
            "good_response": """Caitlin Cannon is a product manager with 7+ years leading discovery, prioritizing backlogs, and tracking KPIs for B2B SaaS analytics products [1]. She has worked with squads, using tools like Jira, Figma, and Mixpanel to align delivery with metrics [1].

**Sources consulted:**
1. CV Caitlin Cannon - cv://cv-02-caitlin-cannon (Relevance: 86%)""",
            "bad_response": "She has a lot of product experience and uses many tools.",
            "reason": "Bad: Lacks specifics and citations, too generic"
        }
    ]

    @classmethod
    def create_system_prompt(cls, language: str = "en", query_complexity: str = "simple") -> str:
        """
        Create complete system prompt with hierarchical structure.

        Args:
            language: Language code (always "en")
            query_complexity: Query complexity level

        Returns:
            Complete system prompt
        """
        task_instructions = cls.TASK_INSTRUCTIONS_EN

        prompt = f"""{cls.META_INSTRUCTIONS}

{cls.DOMAIN_KNOWLEDGE}

{task_instructions}"""

        return prompt

    @classmethod
    def format_few_shot_examples(cls, language: str = "en") -> str:
        """Format few-shot examples for the prompt."""
        relevant_examples = cls.FEW_SHOT_EXAMPLES
        label = "Example" if language == "en" else "Ejemplo"

        formatted = []
        for i, example in enumerate(relevant_examples, 1):
            formatted.append(f"""
**{label} {i}** - {example['name']}:

Available context:
{example['context'].strip()}

Question: {example['question']}

✅ **Correct answer**:
{example['good_response'].strip()}

❌ **Incorrect answer (do NOT do this)**:
{example['bad_response'].strip()}
Reason: {example['reason']}
""")
        return "\n".join(formatted)

    @classmethod
    def create_full_prompt(
            cls,
            context: str,
            chat_history: str,
            question: str,
            language: str = "en",
            query_complexity: str = "simple",
            include_few_shot: bool = True,
    ) -> str:
        """
        Create complete optimized prompt with all enhancements.

        Args:
            context: Retrieved context from documents (pre-formatted with tiers).
            chat_history: Conversation history.
            question: User question.
            language: Language code ("es" or "en").
            query_complexity: Query complexity level.
            include_few_shot: Whether to include few-shot examples.

        Returns:
            Complete formatted prompt.
        """
        system_prompt = cls.create_system_prompt(language, query_complexity)

        few_shot = ""
        if include_few_shot:
            few_shot = f"""
# EXAMPLE ANSWERS (Learn from these)
{cls.format_few_shot_examples(language)}

---
"""

        context_label = "# RETRIEVED CONTEXT"
        context_intro = "The following sources were retrieved for your query (ordered by relevance):"
        history_label = "# CONVERSATION HISTORY"
        question_label = "# USER QUESTION"
        response_label = "# YOUR RESPONSE"
        history_text = chat_history if chat_history else "No prior conversation."

        full_prompt = f"""{system_prompt}

{few_shot}
{context_label}
{context_intro}

{context}

---

{history_label}
{history_text}

---

{question_label}
{question}

---

{response_label}
Provide a well-structured, cited answer following the instructions above:"""

        return full_prompt


class QueryAnalyzer:
    """
    Analyzes queries to determine optimal processing strategy.
    """

    def __init__(self, openai_client=None):
        """
        Initialize QueryAnalyzer

        Args:
            openai_client: Optional OpenAI client (kept for interface compatibility).
        """
        self.openai_client = openai_client

    def detect_language(self, query: str) -> str:
        """
        Detect query language

        Args:
            query: User query

        Returns:
            Language code: always "en"
        """
        return "en"

    @staticmethod
    def classify_complexity(query: str) -> str:
        """
        Classify query complexity to determine processing strategy.

        Args:
            query: User query

        Returns:
            Complexity level: "simple", "moderate", or "complex"
        """
        query_lower = query.lower()

        complex_markers = [
            "compare", "difference", "pros and cons",
            "why", "how does", "process", "implement",
            "better than", "vs", "versus", "explain"
        ]

        moderate_markers = [
            "features", "capabilities", "includes", "offers",
            "types of", "which"
        ]

        for marker in complex_markers:
            if marker in query_lower:
                return "complex"

        for marker in moderate_markers:
            if marker in query_lower:
                return "moderate"

        return "simple"

    @staticmethod
    def augment_short_query(
            query: str,
            chat_history: List[Tuple[str, str]]
    ) -> Tuple[str, bool]:
        """
        Augment short or ambiguous queries with context from history.

        Args:
            query: User query
            chat_history: Recent conversation history

        Returns:
            Tuple of (augmented_query, was_augmented)
        """
        words = query.split()
        if len(words) > 2:
            return query, False

        if not chat_history:
            return query, False

        last_q, last_a = chat_history[-1]

        topics = {
            "profile": "about the candidate profile",
            "experience": "about the professional experience",
            "skills": "about the candidate skills",
            "education": "about the education",
        }

        for topic, augmentation in topics.items():
            if topic.lower() in last_a.lower():
                augmented = f"{query} {augmentation}"
                logger.info(f"Augmented short query: '{query}' → '{augmented}'")
                return augmented, True

        return query, False


class GuardrailValidator:
    """
    Enhanced validator with stricter citation and grounding checks.
    """

    @staticmethod
    def validate_response(
            response: str,
            context: str,
            sources: List[Dict],
            language: str = "en"
    ) -> Dict[str, Any]:
        """
        Validate a response against enhanced guardrails.

        Args:
            response: Generated response.
            context: Context used for generation.
            sources: Source documents.
            language: Expected response language.

        Returns:
            Validation result with pass/fail and detailed issues.
        """
        issues = []

        if not response or len(response.strip()) < 10:
            issues.append("Response too short or empty")

        inline_citations = re.findall(r'\[\d+\]', response)
        if sources and not inline_citations:
            issues.append("Missing inline citations [N]")

        footnote_markers = ["**Sources consulted:**"]
        has_footnotes = any(marker in response for marker in footnote_markers)
        if sources and not has_footnotes:
            issues.append("Missing footnote section with sources")

        if sources and inline_citations:
            cited_numbers = set(int(re.search(r'\d+', c).group()) for c in inline_citations)
            available_numbers = set(range(1, len(sources) + 1))
            unused_sources = available_numbers - cited_numbers
            if unused_sources:
                issues.append(f"Sources not cited: {unused_sources}")

            phantom = cited_numbers - available_numbers
            if phantom:
                issues.append(f"Citations to non-existent sources: {phantom}")

        if context and len(context) > 50:
            common_words = set(context.lower().split()) & set(response.lower().split())
            if len(common_words) / max(len(set(response.lower().split())), 1) > 0.8:
                issues.append("Response appears to copy context without synthesis")

        expected_words = {
            "en": ["the", "is", "are", "of", "and", "to", "a", "in", "for", "with"]
        }

        words = response.lower().split()
        target_words = expected_words.get(language, expected_words["en"])
        lang_ratio = sum(1 for w in words if w in target_words) / max(len(words), 1)

        if lang_ratio < 0.08:
            issues.append(f"Response may not be in expected language ({language})")

        word_count = len(response.split())
        if word_count > 700:
            issues.append(f"Response too long ({word_count} words, max 600)")

        hallucination_phrases = {
            "en": ["in my knowledge", "as far as I know", "in my experience",
                   "I think", "I imagine", "probably without"]
        }

        response_lower = response.lower()
        found_hallucinations = []
        for phrase in hallucination_phrases.get(language, hallucination_phrases["en"]):
            if phrase in response_lower:
                found_hallucinations.append(phrase)

        if found_hallucinations:
            issues.append(f"Hallucination indicators found: {found_hallucinations}")

        score = max(0.0, 1.0 - (len(issues) * 0.15))

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "score": score,
            "inline_citations_count": len(inline_citations),
            "has_footnotes": has_footnotes,
        }

    @staticmethod
    def detect_fabricated_claims(response: str, language: str = "en") -> List[str]:
        """
        Detect potentially fabricated specific claims.

        Args:
            response: Generated response.
            language: Response language.

        Returns:
            List of suspicious claims found.
        """
        suspicious_claims = []

        # Specific claim markers that should ALWAYS have citations
        claim_markers = {
            "en": [
                (r'\d+[\s]*€', "specific price in euros"),
                (r'\d+[\s]*\$', "specific price in dollars"),
                (r'founded in \d{4}', "founding year"),
                (r'costs.*\d+', "cost statement"),
                (r'price.*\d+', "price statement"),
                (r'\d+ employees', "employee count"),
                (r'\d+% of', "specific percentage"),
            ]
        }

        response_lower = response.lower()
        markers = claim_markers.get(language, claim_markers["en"])

        for pattern, claim_type in markers:
            matches = re.findall(pattern, response_lower)
            if matches:
                suspicious_claims.append(f"{claim_type}: {matches}")

        return suspicious_claims


class PromptOptimizer:
    """
    Main class for optimized prompt generation with semantic enhancements.

    Integrates:
    - Query analysis (language, complexity)
    - Context-aware augmentation
    - Hierarchical prompt structuring
    - Enhanced validation
    """

    def __init__(
            self,
            use_few_shot: bool = True,
            validate_output: bool = True,
            auto_augment_short_queries: bool = True,
            openai_client=None,
    ):
        """
        Initialize prompt optimizer.

        Args:
            use_few_shot: Whether to include few-shot examples.
            validate_output: Whether to validate outputs.
            auto_augment_short_queries: Whether to auto-augment short queries.
            openai_client: Optional OpenAI client for LLM-based language detection.
        """
        self.use_few_shot = use_few_shot
        self.validate_output = validate_output
        self.auto_augment_short_queries = auto_augment_short_queries
        self.validator = GuardrailValidator()
        self.analyzer = QueryAnalyzer(openai_client=openai_client)

        logger.info(
            f"PromptOptimizer initialized: few_shot={use_few_shot}, "
            f"validate={validate_output}, auto_augment={auto_augment_short_queries}, "
            f"language_mode=english_only"
        )

    def create_prompt(
            self,
            context: str,
            chat_history: str,
            question: str,
            chat_history_list: Optional[List[Tuple[str, str]]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Create optimized prompt with automatic analysis and adaptation.

        Args:
            context: Retrieved context (should be pre-formatted with quality tiers).
            chat_history: Formatted conversation history string.
            question: User question.
            chat_history_list: Optional list of (question, answer) tuples for augmentation.

        Returns:
            Tuple of (optimized_prompt, analysis_metadata)
        """
        language = self.analyzer.detect_language(question)
        complexity = self.analyzer.classify_complexity(question)

        original_question = question
        was_augmented = False
        if self.auto_augment_short_queries and chat_history_list:
            question, was_augmented = self.analyzer.augment_short_query(
                question, chat_history_list
            )

        prompt = PromptTemplate.create_full_prompt(
            context=context,
            chat_history=chat_history,
            question=question,
            language=language,
            query_complexity=complexity,
            include_few_shot=self.use_few_shot,
        )

        metadata = {
            "language": language,
            "complexity": complexity,
            "original_question": original_question,
            "augmented_question": question if was_augmented else None,
            "was_augmented": was_augmented,
            "prompt_length": len(prompt),
        }

        logger.info(
            f"Created prompt: lang={language}, complexity={complexity}, "
            f"augmented={was_augmented}, length={len(prompt)}"
        )

        return prompt, metadata

    def validate_response(
            self,
            response: str,
            context: str,
            sources: List[Dict],
            language: str = "en",
    ) -> Dict[str, Any]:
        """
        Validate generated response with enhanced checks.

        Args:
            response: Generated response.
            context: Context used.
            sources: Source documents.
            language: Expected language.

        Returns:
            Comprehensive validation results.
        """
        if not self.validate_output:
            return {
                "passed": True,
                "issues": [],
                "score": 1.0,
                "validation_skipped": True
            }

        validation = self.validator.validate_response(
            response, context, sources, language
        )

        fabricated_claims = self.validator.detect_fabricated_claims(response, language)

        result = {
            **validation,
            "fabricated_claims": fabricated_claims,
            "has_fabricated_claims": len(fabricated_claims) > 0,
        }

        if fabricated_claims:
            result["issues"].append(f"Potential fabricated claims: {fabricated_claims}")
            result["passed"] = False
            result["score"] = max(0.0, result["score"] - 0.3)

        logger.info(
            f"Validation: passed={result['passed']}, score={result['score']:.2f}, "
            f"issues={len(result['issues'])}"
        )

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics and configuration."""
        return {
            "use_few_shot": self.use_few_shot,
            "validate_output": self.validate_output,
            "auto_augment_short_queries": self.auto_augment_short_queries,
            "version": "2.0",
            "features": [
                "english_only",
                "chain_of_thought",
                "query_augmentation",
                "tiered_context",
                "enhanced_validation",
                "confidence_expression",
            ]
        }


GuardrailValidator = GuardrailValidator
OutputSchema = None
