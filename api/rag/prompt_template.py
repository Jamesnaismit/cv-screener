"""
Prompt template definitions and rendering helpers.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


class PromptTemplate:
    """Prompt pieces and render helpers for CV Q&A."""

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
        """Create complete system prompt."""
        _ = query_complexity  # kept for compatibility/future use
        task_instructions = cls.TASK_INSTRUCTIONS_EN

        return f"""{cls.META_INSTRUCTIONS}

{cls.DOMAIN_KNOWLEDGE}

{task_instructions}"""

    @classmethod
    def format_few_shot_examples(cls, language: str = "en") -> str:
        """Format few-shot examples for the prompt."""
        relevant_examples = cls.FEW_SHOT_EXAMPLES
        label = "Example"

        formatted: List[str] = []
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
            language: Language code (always "en").
            query_complexity: Query complexity level (kept for compatibility).
            include_few_shot: Whether to include few-shot examples.
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

        return f"""{system_prompt}

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
