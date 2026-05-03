from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import json
import time
import random

load_dotenv()
client = OpenAI()


# ─────────────────────────────────────────────────────────────────────────────
# Rate-limit retry helper
# ─────────────────────────────────────────────────────────────────────────────

def _with_retry(fn, max_attempts: int = 6, base_delay: float = 3.0):
    """
    Call fn(); if a 429 / rate_limit_exceeded error is raised, wait with
    exponential back-off + jitter and try again (up to max_attempts times).
    All other exceptions propagate immediately.
    """
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            msg = str(exc)
            is_rate_limit = (
                "rate_limit_exceeded" in msg
                or "429" in msg
                or "RateLimitError" in type(exc).__name__
            )
            if is_rate_limit and attempt < max_attempts - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0.5, 2.0)
                time.sleep(delay)
                continue
            raise


# ─────────────────────────────────────────────────────────────────────────────
# Parallelism helper
# ─────────────────────────────────────────────────────────────────────────────

def _par(*fns):
    """Run callables in parallel threads with rate-limit retry; return results in submission order."""
    with ThreadPoolExecutor(max_workers=len(fns)) as ex:
        futures = [ex.submit(_with_retry, fn) for fn in fns]
        return [f.result() for f in futures]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(
    job_description: str,
    resume: str,
    recruiter_profile: str = "",
    on_step=None,
) -> dict:
    """
    Full 10-phase pipeline with parallel execution.

    Phases:
      0  – Parallel extraction (resume + JD + optional recruiter)
      1  – Parallel fit score + analysis
      2  – Cover letter strategy
      3  – Parallel CL draft + recruiter commonalities
      4  – CL polish + enforce
      5  – Parallel LI strategy + analysis polish + company research
      6  – Parallel LI draft + resume draft
      7  – Parallel LI polish + resume critique
      8  – Resume refinement
      9  – Parallel fact verify + interview prep + follow-up email

    Args:
        on_step: Optional callable(str) called with a progress message before each phase.
    """

    def _log(msg: str):
        if on_step:
            on_step(msg)

    # ── Phase 0: parallel extraction ──────────────────────────────────────────
    _log("📄 Extracting resume and job intelligence…")
    if recruiter_profile and recruiter_profile.strip():
        resume_intel, jd_intel, recruiter_intel = _par(
            lambda: _extract_resume(resume),
            lambda: _extract_job_description(job_description),
            lambda: _extract_recruiter_profile(recruiter_profile),
        )
    else:
        resume_intel, jd_intel = _par(
            lambda: _extract_resume(resume),
            lambda: _extract_job_description(job_description),
        )
        recruiter_intel = None

    # ── Phase 1: parallel fit score + analysis ────────────────────────────────
    _log("📊 Scoring fit and analyzing requirements…")
    fit_result, raw_analysis = _par(
        lambda: _compute_fit_score(job_description, resume, jd_intel, resume_intel),
        lambda: _generate_analysis(job_description, resume, jd_intel, resume_intel),
    )
    analysis = raw_analysis
    analysis["fit_score"]    = fit_result["fit_score"]
    analysis["fit_scoring"]  = fit_result["scoring"]
    analysis["fit_rationale"] = fit_result.get("fit_rationale", "")

    # ── Phase 2: cover letter strategy ───────────────────────────────────────
    _log("🎯 Building cover letter strategy…")
    cl_strategy  = _generate_cover_letter_strategy(job_description, resume, analysis, jd_intel)
    company_name = cl_strategy.get("company_name", "the company")

    # ── Phase 3: parallel CL draft + recruiter analysis ──────────────────────
    _log("✍️  Drafting cover letter" + (" and analyzing recruiter profile…" if recruiter_intel else "…"))
    if recruiter_intel:
        cl_draft, recruiter_analysis = _par(
            lambda: _generate_cover_letter(job_description, resume, analysis, cl_strategy, company_name),
            lambda: _analyze_recruiter_commonalities(recruiter_intel, resume_intel),
        )
    else:
        cl_draft = _generate_cover_letter(job_description, resume, analysis, cl_strategy, company_name)
        recruiter_analysis = None

    # ── Phase 4: CL polish + enforce ──────────────────────────────────────────
    _log("✨ Polishing cover letter…")
    cl_polished = _polish_cover_letter(cl_draft, cl_strategy, company_name)
    cl_final    = _enforce_cover_letter_format(cl_polished, company_name)

    # ── Phase 5: parallel LI strategy + analysis polish + company research ───
    _log("🔍 Building LinkedIn strategy, polishing analysis, and researching company…")
    li_strategy, polished_analysis, company_research = _par(
        lambda: _generate_linkedin_strategy(job_description, resume, analysis, cl_strategy, recruiter_analysis),
        lambda: _improve_analysis(analysis),
        lambda: _generate_company_research(job_description, jd_intel),
    )

    # ── Phase 6: parallel LI draft + resume draft ────────────────────────────
    _log("📝 Drafting LinkedIn message and tailoring resume…")
    li_draft, tailored_resume_draft = _par(
        lambda: _generate_linkedin_message(job_description, resume, analysis, li_strategy, recruiter_analysis),
        lambda: _generate_tailored_resume(job_description, resume, polished_analysis, cl_strategy, jd_intel, resume_intel),
    )

    # ── Phase 7: parallel LI polish + resume critique ────────────────────────
    _log("🔬 Polishing LinkedIn message and critiquing resume…")
    li_final, optimization = _par(
        lambda: _polish_linkedin_message(li_draft, li_strategy, recruiter_analysis),
        lambda: _generate_resume_optimization_prompt(job_description, resume, tailored_resume_draft, polished_analysis, jd_intel),
    )

    # ── Phase 8: resume refinement ────────────────────────────────────────────
    _log("⚡ Refining resume with optimization brief…")
    tailored_resume = _refine_tailored_resume(
        job_description, resume, tailored_resume_draft, optimization, resume_intel
    )

    # ── Phase 9: parallel fact verify + interview prep + follow-up email ──────
    _log("🛡️  Verifying accuracy, generating interview prep and follow-up email…")
    fact_check, interview_prep, follow_up_email = _par(
        lambda: _verify_factual_accuracy(cl_final, li_final, tailored_resume, resume, resume_intel),
        lambda: _generate_interview_prep(job_description, resume, polished_analysis, jd_intel, resume_intel),
        lambda: _generate_follow_up_email(cl_strategy, cl_final, company_name),
    )

    cl_verified = fact_check.get("cleaned_cover_letter", cl_final)
    li_verified = fact_check.get("cleaned_linkedin_message", li_final)

    return {
        **polished_analysis,
        "cover_letter":          cl_verified,
        "cover_letter_strategy": cl_strategy,
        "linkedin_message":      li_verified,
        "linkedin_strategy":     li_strategy,
        "recruiter_analysis":    recruiter_analysis,
        "tailored_resume":       tailored_resume,
        "resume_optimization":   optimization,
        "fact_check_report":     fact_check,
        "company_research":      company_research,
        "interview_prep":        interview_prep,
        "follow_up_email":       follow_up_email,
        "resume_intel":          resume_intel,
        "jd_intel":              jd_intel,
    }


# =============================================================================
# PHASE 0 — EXTRACTION LAYER
# =============================================================================

def _extract_resume(resume: str) -> dict:
    """
    Deep structured extraction of every verifiable fact from the candidate's resume.

    The `quantified_achievements` list contains verbatim metric quotes — every
    downstream generation step is given this list and told these are the ONLY
    numbers it may use, which is the primary anti-hallucination mechanism.
    """
    response = client.chat.completions.create(
        model="gpt-4.1",
        response_format={"type": "json_object"},
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precision data-extraction engine. "
                    "Extract facts that are EXPLICITLY STATED in the resume. "
                    "Do NOT infer, assume, or invent anything. "
                    "If a field cannot be found, use empty string or empty list. "
                    "Return ONLY valid JSON."
                ),
            },
            {
                "role": "user",
                "content": f"""
Extract every verifiable fact from this resume.

CRITICAL: `quantified_achievements` must contain VERBATIM quotes of every specific metric,
number, percentage, dollar amount, team size, or measurable result anywhere in the resume.
These are the ONLY numbers downstream AI steps are permitted to use.

Return JSON with EXACTLY this structure:
{{
  "candidate_name": "string",
  "contact": {{
    "email": "string",
    "phone": "string",
    "linkedin": "string",
    "location": "string"
  }},
  "total_years_experience": number,
  "seniority_level": "string — e.g. Senior IC, Director, VP",
  "career_arc": "string — 1–2 sentences describing the trajectory objectively",
  "industries": ["string"],
  "functions": ["string — e.g. Product, Engineering, Sales Ops, Finance"],
  "experience": [
    {{
      "company": "string — exact name",
      "location": "string",
      "roles": [
        {{
          "title": "string — exact title",
          "dates": "string — exact date range",
          "duration_months": number,
          "responsibilities": ["string — verbatim or close paraphrase"],
          "achievements": ["string — verbatim accomplishments"]
        }}
      ]
    }}
  ],
  "quantified_achievements": [
    {{
      "raw_text": "string — verbatim quote containing a number or metric",
      "metric_type": "string — revenue | headcount | growth | cost | time | percentage | other",
      "company": "string"
    }}
  ],
  "skills": {{
    "technical": ["string"],
    "domain": ["string"],
    "tools_and_platforms": ["string"],
    "methodologies": ["string"],
    "soft_skills": ["string"]
  }},
  "education": [
    {{
      "school": "string",
      "degree": "string",
      "field": "string",
      "graduation_year": "string"
    }}
  ],
  "certifications": ["string"],
  "keywords": ["string — all notable ATS-relevant terms"]
}}

RESUME:
{resume}
""",
            },
        ],
    )
    return json.loads(response.choices[0].message.content)


def _extract_job_description(job_description: str) -> dict:
    """
    Deep structured extraction from the job posting.

    Separates must-haves from nice-to-haves, identifies ATS keywords, culture
    signals, and the vocabulary every generation step should mirror.
    """
    response = client.chat.completions.create(
        model="gpt-4.1",
        response_format={"type": "json_object"},
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precision job-description analyst. "
                    "Extract only what is explicitly stated. "
                    "Distinguish hard requirements from preferences. "
                    "Return ONLY valid JSON."
                ),
            },
            {
                "role": "user",
                "content": f"""
Extract every signal from this job description.

Return JSON with EXACTLY this structure:
{{
  "company_name": "string",
  "role_title": "string",
  "department": "string",
  "seniority_level": "string",
  "team_context": "string — team size, reporting structure, scope if mentioned",
  "industry": "string",
  "company_stage": "string — e.g. Series B, public, Fortune 500",
  "company_culture_signals": ["string — specific phrases revealing culture or values"],
  "must_have_requirements": ["string — explicitly required qualifications"],
  "nice_to_have_requirements": ["string — preferred but not required"],
  "key_responsibilities": ["string — core duties in priority order"],
  "required_skills": {{
    "technical": ["string"],
    "domain": ["string"],
    "tools_and_platforms": ["string"],
    "soft_skills": ["string"]
  }},
  "success_metrics": ["string — what success looks like in this role"],
  "ats_keywords": ["string — highest-priority ATS terms"],
  "vocabulary_to_mirror": ["string — specific phrases worth mirroring verbatim"],
  "tone": "string — JD writing tone",
  "compensation": "string — salary range or 'not mentioned'",
  "location_requirements": "string",
  "red_flags": ["string — unusual demands, vague scope, concerns"],
  "value_proposition": "string — what makes this role compelling"
}}

JOB DESCRIPTION:
{job_description}
""",
            },
        ],
    )
    return json.loads(response.choices[0].message.content)


def _extract_recruiter_profile(profile_text: str) -> dict:
    """Structure the recruiter's LinkedIn profile for field-level commonality matching."""
    response = client.chat.completions.create(
        model="gpt-4.1",
        response_format={"type": "json_object"},
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract only what is explicitly stated in this LinkedIn profile. "
                    "Return ONLY valid JSON."
                ),
            },
            {
                "role": "user",
                "content": f"""
Extract every verifiable detail from this recruiter's LinkedIn profile.

Return JSON:
{{
  "name": "string",
  "current_title": "string",
  "current_company": "string",
  "location": "string",
  "about_summary": "string",
  "education": [{{"school": "string", "degree": "string", "field": "string", "dates": "string"}}],
  "experience": [{{"company": "string", "title": "string", "dates": "string", "description": "string"}}],
  "all_companies_worked_at": ["string"],
  "all_schools_attended": ["string"],
  "skills_and_endorsements": ["string"],
  "interests_and_causes": ["string"],
  "locations_lived_or_worked": ["string"],
  "industries": ["string"],
  "career_specializations": ["string"]
}}

PROFILE:
{profile_text}
""",
            },
        ],
    )
    return json.loads(response.choices[0].message.content)


# =============================================================================
# PHASE 1 — FIT SCORE
# =============================================================================

def _compute_fit_score(
    job_description: str,
    resume: str,
    jd_intel: dict,
    resume_intel: dict,
) -> dict:
    """
    Rubric-based fit score with five weighted dimensions.

    Python re-computes the weighted average as a safeguard against model arithmetic.
    Low temperature (0.2) keeps scoring consistent across runs.
    """
    must_haves      = jd_intel.get("must_have_requirements", [])
    jd_tools        = jd_intel.get("required_skills", {}).get("tools_and_platforms", [])
    candidate_tools = (
        resume_intel.get("skills", {}).get("tools_and_platforms", [])
        + resume_intel.get("skills", {}).get("technical", [])
    )

    response = client.chat.completions.create(
        model="gpt-4.1",
        response_format={"type": "json_object"},
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior technical recruiter scoring candidate–role fit. "
                    "You are calibrated and honest — you do NOT inflate scores. "
                    "A 10 means near-perfect match on every hard requirement. "
                    "A 5 means relevant background but significant gaps remain. "
                    "A 3 or below means fundamental misalignment on hard requirements. "
                    "Score each dimension INDEPENDENTLY before computing the weighted final. "
                    "Return ONLY valid JSON."
                ),
            },
            {
                "role": "user",
                "content": f"""
Score this candidate against the role using the rubric below.

EXTRACTED MUST-HAVE REQUIREMENTS (from JD):
{json.dumps(must_haves, indent=2)}

CANDIDATE TOOLS & SKILLS (from resume):
{json.dumps(candidate_tools, indent=2)}

CANDIDATE PROFILE:
- Years of experience: {resume_intel.get("total_years_experience", "unknown")}
- Seniority: {resume_intel.get("seniority_level", "unknown")}
- Industries: {json.dumps(resume_intel.get("industries", []))}
- Functions: {json.dumps(resume_intel.get("functions", []))}

ROLE REQUIREMENTS:
- Required tools: {json.dumps(jd_tools)}
- Expected seniority: {jd_intel.get("seniority_level", "unknown")}
- Industry: {jd_intel.get("industry", "unknown")}

RUBRIC (score each 1–10 independently):

1. must_have_requirements (weight 0.35)
   — Does the candidate's DEMONSTRATED experience satisfy each hard requirement?
     Deduct 2+ points per unchecked must-have. No credit for adjacent or implied experience.

2. technical_skill_overlap (weight 0.25)
   — What share of the JD's specific tools/technologies appear in the candidate's profile?
     Score proportionally (e.g. 6 of 10 tools present = ~6/10).

3. years_and_seniority (weight 0.20)
   — Does total relevant experience and seniority level match the role's requirements?
     Penalise if under- OR significantly over-qualified.

4. leadership_and_scope (weight 0.10)
   — Does demonstrated team size, budget ownership, and org influence match the role?

5. domain_and_industry_fit (weight 0.10)
   — Does the candidate's industry and domain background align with the company's context?

RULES:
- Score each dimension before computing the weighted final.
- Round final to the nearest integer. Do not round up to flatter the candidate.

Return JSON:
{{
  "scoring": {{
    "must_have_requirements":  {{"score": 1-10, "rationale": "1 sentence"}},
    "technical_skill_overlap": {{"score": 1-10, "rationale": "1 sentence"}},
    "years_and_seniority":     {{"score": 1-10, "rationale": "1 sentence"}},
    "leadership_and_scope":    {{"score": 1-10, "rationale": "1 sentence"}},
    "domain_and_industry_fit": {{"score": 1-10, "rationale": "1 sentence"}}
  }},
  "fit_score": "weighted integer 1-10",
  "fit_rationale": "2 sentences: what drives the score up and what holds it back"
}}

JOB DESCRIPTION:
{job_description}

RESUME:
{resume}
""",
            },
        ],
    )
    data = json.loads(response.choices[0].message.content)

    weights = {
        "must_have_requirements":  0.35,
        "technical_skill_overlap": 0.25,
        "years_and_seniority":     0.20,
        "leadership_and_scope":    0.10,
        "domain_and_industry_fit": 0.10,
    }
    scoring  = data.get("scoring", {})
    weighted = sum(
        scoring.get(dim, {}).get("score", 5) * w for dim, w in weights.items()
    )
    data["fit_score"] = max(1, min(10, round(weighted)))
    return data


# =============================================================================
# PHASE 1 — ANALYSIS
# =============================================================================

def _generate_analysis(
    job_description: str,
    resume: str,
    jd_intel: dict,
    resume_intel: dict,
) -> dict:
    """
    Produce key requirements, strengths, gaps, and resume bullets.
    Fit score is added by the caller from _compute_fit_score.

    Each strength must cite specific resume evidence.
    Each bullet may only use numbers from the quantified_achievements list.
    """
    quant_text = "\n".join(
        f"  • [{a.get('company','')}] {a.get('raw_text','')}"
        for a in resume_intel.get("quantified_achievements", [])
    ) or "  (none extracted)"

    response = client.chat.completions.create(
        model="gpt-4.1",
        response_format={"type": "json_object"},
        temperature=0.6,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an elite job application strategist. "
                    "Every strength must cite a specific, verifiable fact from the resume — "
                    "no vague claims like 'strong communicator'. "
                    "Every resume bullet must use ONLY numbers from the quantified achievements list provided — "
                    "never invent metrics. "
                    "Gaps must be honest and specific — frame each as a concrete, addressable delta. "
                    "Return ONLY valid JSON."
                ),
            },
            {
                "role": "user",
                "content": f"""
Analyze the job description and resume. Use the extracted intel below for precision.

MUST-HAVE REQUIREMENTS (use as the gap checklist):
{json.dumps(jd_intel.get("must_have_requirements", []), indent=2)}

NICE-TO-HAVES:
{json.dumps(jd_intel.get("nice_to_have_requirements", []), indent=2)}

VERIFIED QUANTIFIED ACHIEVEMENTS (ONLY these numbers may appear in resume_bullets):
{quant_text}

CANDIDATE SKILLS:
{json.dumps(resume_intel.get("skills", {}), indent=2)}

CAREER ARC:
{resume_intel.get("career_arc", "")}

Return JSON:
{{
  "key_requirements": ["role's actual hard demands in priority order"],
  "strengths": ["Each must follow format: '[Specific evidence from resume] → [How it maps to JD requirement]'"],
  "gaps": ["Specific gap vs. must-have requirements — frame as an addressable delta, not a judgment"],
  "resume_bullets": ["5 strong, quantified, role-relevant bullets — use ONLY numbers from the list above"]
}}

JOB DESCRIPTION:
{job_description}

RESUME:
{resume}
""",
            },
        ],
    )
    return json.loads(response.choices[0].message.content)


# =============================================================================
# PHASE 2 — COVER LETTER STRATEGY
# =============================================================================

def _generate_cover_letter_strategy(
    job_description: str,
    resume: str,
    analysis: dict,
    jd_intel: dict | None = None,
) -> dict:
    """
    Design a precise writing brief before a single word of the letter is drafted.

    Forcing a strategy phase prevents the model from defaulting to generic
    cover letter patterns and grounds every sentence in JD-specific signals.
    """
    response = client.chat.completions.create(
        model="gpt-4.1",
        response_format={"type": "json_object"},
        temperature=0.6,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior executive career strategist briefing a ghostwriter. "
                    "Your job is NOT to write the letter — design the strategy FOR it. "
                    "Be specific and opinionated. Vague guidance produces generic letters. "
                    "Return ONLY valid JSON."
                ),
            },
            {
                "role": "user",
                "content": f"""
Design an optimal cover letter strategy for this candidate applying to this role.

EXTRACTED JD SIGNALS:
- Company: {(jd_intel or {}).get("company_name", "")}
- Culture signals: {json.dumps((jd_intel or {}).get("company_culture_signals", []))}
- Tone: {(jd_intel or {}).get("tone", "")}
- ATS keywords: {json.dumps((jd_intel or {}).get("ats_keywords", []))}
- Vocabulary to mirror: {json.dumps((jd_intel or {}).get("vocabulary_to_mirror", []))}
- Value proposition: {(jd_intel or {}).get("value_proposition", "")}

CANDIDATE STRENGTHS:
{json.dumps(analysis.get("strengths", []), indent=2)}

Return JSON:
{{
  "company_name": "string — exactly as it should appear in the greeting",
  "company_culture_signals": "string — what the JD reveals about culture, tone, values",
  "candidate_narrative_angle": "string — the single most compelling story this candidate can tell for THIS role. One sentence. Be ruthlessly specific.",
  "opening_hook": "string — the opening SENTENCE of the letter body (after the greeting). Must convey specific measurable impact in the first 10 words. FAILURE MODES: 'With over X years of experience…', 'I am writing to express…', 'Having worked in X for Y years…'. SUCCESS: reference a concrete result, decision, or moment that is instantly relevant to what this company needs right now.",
  "key_themes": [
    "exactly 3 strings — the 3 most powerful candidate achievements tied directly to a specific challenge or phrase in the JD"
  ],
  "company_language_to_mirror": [
    "4–6 specific words or phrases taken verbatim from the JD"
  ],
  "tone_guidance": "string — the EXACT tone to strike. Be specific: e.g. 'Confident and direct, like a peer, not an applicant. No deference. No enthusiasm-signaling.'",
  "length_and_format": "string — e.g. '3 tight paragraphs, under 260 words, no bullets'",
  "what_to_avoid": [
    "specific phrases, approaches, or topics that would hurt this application"
  ]
}}

JOB DESCRIPTION:
{job_description}

RESUME:
{resume}

CANDIDATE ANALYSIS:
{json.dumps(analysis, indent=2)}
""",
            },
        ],
    )
    return json.loads(response.choices[0].message.content)


# =============================================================================
# PHASE 3 — DRAFT COVER LETTER
# =============================================================================

def _generate_cover_letter(
    job_description: str,
    resume: str,
    analysis: dict,
    strategy: dict,
    company_name: str,
) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0.75,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an elite ghostwriter for senior executives. "
                    "You write cover letters that get responses — not compliments. "
                    "You follow the strategy brief with surgical precision. "
                    "You never use clichés, generic openers, or HR-speak. "
                    "You never begin a sentence with 'I'. "
                    "Every sentence earns its place or it's cut."
                ),
            },
            {
                "role": "user",
                "content": f"""
Write a cover letter using this strategy brief as your exact instructions.

STRATEGY BRIEF:
{json.dumps(strategy, indent=2)}

CANDIDATE STRENGTHS TO DRAW FROM (cite specific evidence):
{json.dumps(analysis.get("strengths", []), indent=2)}

TOP RESUME BULLETS (use or adapt with real numbers only):
{json.dumps(analysis.get("resume_bullets", []), indent=2)}

FORMAT RULES (non-negotiable):
- First line exactly: Hello {company_name} Recruiting Team,
- Blank line, then body starting with the opening_hook from the brief
- The opening hook must be the exact first sentence — do not soften or generalize it
- Cover all 3 key_themes with specific evidence from the resume
- Weave company_language_to_mirror naturally — never force it
- Match tone_guidance and length_and_format exactly
- Do NOT include a subject line, date, or address block
- Last two lines exactly: Thanks, / Max
- Return ONLY the cover letter text

JOB DESCRIPTION (for reference):
{job_description}

RESUME (for reference — only use facts from here):
{resume}
""",
            },
        ],
    )
    return response.choices[0].message.content.strip()


# =============================================================================
# PHASE 4 — POLISH COVER LETTER
# =============================================================================

def _polish_cover_letter(draft: str, strategy: dict, company_name: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0.45,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a ruthless editor who has reviewed 50,000 cover letters. "
                    "You cut every word that doesn't earn its place. "
                    "You sharpen language without adding length. "
                    "You make prose feel human, confident, and specific — never like AI output. "
                    "You never soften strong opening hooks."
                ),
            },
            {
                "role": "user",
                "content": f"""
Polish this cover letter. Use the strategy brief as a quality checklist.

STRATEGY BRIEF:
- Tone: {strategy.get("tone_guidance")}
- Format: {strategy.get("length_and_format")}
- Avoid: {json.dumps(strategy.get("what_to_avoid", []))}
- Language to mirror: {json.dumps(strategy.get("company_language_to_mirror", []))}

EDITS MUST:
1. Sharpen every sentence — cut anything that doesn't move the narrative forward
2. Preserve the opening hook EXACTLY — do not water it down
3. Ensure all 3 key themes land with specific evidence, not vague claims
4. Eliminate any residual HR-speak, enthusiasm-signaling, or passive voice
5. Confirm tone matches the brief exactly
6. Ask: "Does this sound like a confident human peer or an AI job applicant?" Fix anything that sounds like the latter

PRESERVE EXACTLY:
- First line: Hello {company_name} Recruiting Team,
- Last two lines: Thanks, / Max

Return ONLY the final polished letter.

DRAFT:
{draft}
""",
            },
        ],
    )
    return response.choices[0].message.content.strip()


def _enforce_cover_letter_format(text: str, company_name: str) -> str:
    """Deterministic greeting and sign-off enforcement. Runs after the polish pass."""
    lines = [l.rstrip() for l in text.strip().splitlines()]
    greeting = f"Hello {company_name} Recruiting Team,"
    signoff_kw = {"best,", "sincerely,", "regards,", "thanks,", "thank you,", "warm regards,"}

    while lines and (
        lines[0].lower().startswith("dear ")
        or lines[0].lower().startswith("hello ")
        or lines[0].lower().startswith("hi ")
    ):
        lines.pop(0)
    while lines and lines[0] == "":
        lines.pop(0)

    lines = [greeting, ""] + lines

    while lines and lines[-1].strip().lower() in {"", "max", "[your name]", "your name"}:
        lines.pop()
    while lines and lines[-1].strip().lower() in signoff_kw:
        lines.pop()

    lines += ["", "Thanks,", "Max"]
    return "\n".join(lines)


# =============================================================================
# RECRUITER COMMONALITY ANALYSIS
# =============================================================================

def _analyze_recruiter_commonalities(
    recruiter_intel: dict,
    resume_intel: dict,
) -> dict:
    """
    Field-level commonality matching between the recruiter and the candidate.

    Operates on pre-extracted structured intel so comparisons are precise
    (school vs school, company vs company) rather than fuzzy text matching.
    """
    candidate_schools   = [e.get("school", "") for e in resume_intel.get("education", [])]
    candidate_companies = [exp.get("company", "") for exp in resume_intel.get("experience", [])]
    candidate_location  = resume_intel.get("contact", {}).get("location", "")
    candidate_skills    = (
        resume_intel.get("skills", {}).get("technical", [])
        + resume_intel.get("skills", {}).get("domain", [])
        + resume_intel.get("skills", {}).get("methodologies", [])
    )

    response = client.chat.completions.create(
        model="gpt-4.1",
        response_format={"type": "json_object"},
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a relationship intelligence analyst. "
                    "Find every genuine, specific overlap between two professionals. "
                    "STRICT RULE: Only surface overlaps that appear in BOTH sets of fields. "
                    "Vague commonalities ('both work in tech') are useless — be specific. "
                    "Return ONLY valid JSON."
                ),
            },
            {
                "role": "user",
                "content": f"""
Compare these fields to find genuine overlaps for use as personalization hooks.

CANDIDATE:
- Schools: {json.dumps(candidate_schools)}
- Companies: {json.dumps(candidate_companies)}
- Location: {candidate_location}
- Skills/domains: {json.dumps(candidate_skills)}
- Industries: {json.dumps(resume_intel.get("industries", []))}
- Career arc: {resume_intel.get("career_arc", "")}

RECRUITER:
- Schools: {json.dumps(recruiter_intel.get("all_schools_attended", []))}
- Companies: {json.dumps(recruiter_intel.get("all_companies_worked_at", []))}
- Location: {recruiter_intel.get("location", "")}
- Skills: {json.dumps(recruiter_intel.get("skills_and_endorsements", []))}
- Interests: {json.dumps(recruiter_intel.get("interests_and_causes", []))}
- Industries: {json.dumps(recruiter_intel.get("industries", []))}
- Current role: {recruiter_intel.get("current_title", "")} at {recruiter_intel.get("current_company", "")}
- About: {recruiter_intel.get("about_summary", "")}

Return JSON:
{{
  "shared_schools": [{{"school": "string", "context": "string", "hook": "how to reference naturally"}}],
  "shared_employers": [{{"company": "string", "context": "string", "hook": "how to reference naturally"}}],
  "shared_locations": [{{"location": "string", "hook": "string"}}],
  "shared_interests_or_domains": [{{"topic": "string", "hook": "string"}}],
  "career_parallels": [{{"parallel": "string", "hook": "string"}}],
  "other_commonalities": [{{"detail": "string", "hook": "string"}}],
  "top_hook": "string — the single most specific and compelling overlap to lead with",
  "secondary_hooks": ["up to 2 additional hooks"],
  "personalization_brief": "2–3 sentences: how to use these hooks so the message feels warm, not researched"
}}
""",
            },
        ],
    )
    return json.loads(response.choices[0].message.content)


# =============================================================================
# PHASE 5 — LINKEDIN STRATEGY
# =============================================================================

def _generate_linkedin_strategy(
    job_description: str,
    resume: str,
    analysis: dict,
    cl_strategy: dict,
    recruiter_analysis: dict | None = None,
) -> dict:
    personalization_section = ""
    if recruiter_analysis and recruiter_analysis.get("top_hook"):
        personalization_section = f"""
RECRUITER PERSONALIZATION:
Lead hook: {recruiter_analysis.get("top_hook")}
Secondary hooks: {json.dumps(recruiter_analysis.get("secondary_hooks", []))}
Execution: {recruiter_analysis.get("personalization_brief")}

The opening_line MUST naturally incorporate the top hook — reference the shared
connection in a way that feels like a warm intro, not a calculated name-drop.
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        response_format={"type": "json_object"},
        temperature=0.6,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a LinkedIn outreach specialist who knows what gets responses. "
                    "Design a writing brief for a LinkedIn InMail — NOT the message itself. "
                    "Platform reality: recruiters see 50+ messages/day. Generic = ignored. "
                    "The only messages that get responses are specific, peer-level, and short. "
                    "Return ONLY valid JSON."
                ),
            },
            {
                "role": "user",
                "content": f"""
Design the optimal LinkedIn outreach brief for this candidate.

COMPANY: {cl_strategy.get("company_name")}
CULTURE: {cl_strategy.get("company_culture_signals")}
CANDIDATE ANGLE: {cl_strategy.get("candidate_narrative_angle")}
{personalization_section}
Return JSON:
{{
  "platform_context": "string — format guidance (e.g. 'Short InMail under 90 words, peer-to-peer tone')",
  "opening_line": "string — the exact first sentence. MUST reference something specific from the JD or the shared connection. MUST NOT be: 'I came across your role', 'I noticed you're hiring', or any generic opener. The reader must feel like you wrote this specifically for them.",
  "value_hook": "string — one sentence on the specific, concrete value this candidate brings to THIS company right now, grounded in a real achievement",
  "conversation_angle": "string — what makes this feel like a peer reaching out, not an applicant applying",
  "cta": "string — a single, low-friction call to action",
  "tone_guidance": "string — exact tone",
  "max_words": 90,
  "what_to_avoid": ["phrases that make LinkedIn messages feel generic, desperate, or AI-generated"]
}}

JOB DESCRIPTION:
{job_description}

CANDIDATE STRENGTHS:
{json.dumps(analysis.get("strengths", []), indent=2)}
""",
            },
        ],
    )
    return json.loads(response.choices[0].message.content)


# =============================================================================
# PHASE 6 — DRAFT LINKEDIN MESSAGE
# =============================================================================

def _generate_linkedin_message(
    job_description: str,
    resume: str,
    analysis: dict,
    strategy: dict,
    recruiter_analysis: dict | None = None,
) -> str:
    personalization_block = ""
    if recruiter_analysis and recruiter_analysis.get("top_hook"):
        personalization_block = f"""
PERSONALIZATION HOOKS:
Primary (lead with this): {recruiter_analysis.get("top_hook")}
Secondary (weave in if space allows): {json.dumps(recruiter_analysis.get("secondary_hooks", []))}
Execution: {recruiter_analysis.get("personalization_brief")}

The hook must feel like a genuine observation — not like you researched their profile to game the opener.
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0.75,
        messages=[
            {
                "role": "system",
                "content": (
                    "You write LinkedIn messages that get responses. "
                    "You follow the brief exactly. Every word is deliberate. "
                    "You never use generic openers, corporate speak, or AI-sounding phrases. "
                    "The message must read like it was written by a sharp, confident human "
                    "who is interested in a conversation — not desperate for a job."
                ),
            },
            {
                "role": "user",
                "content": f"""
Write a LinkedIn outreach message using this brief.

STRATEGY BRIEF:
{json.dumps(strategy, indent=2)}
{personalization_block}
CANDIDATE STRENGTHS:
{json.dumps(analysis.get("strengths", []), indent=2)}

RULES:
- Start with the opening_line from the brief (adapt to incorporate personalization hook if provided)
- Include the value_hook grounded in a real, specific achievement from the resume
- Apply the conversation_angle
- End with the CTA
- Stay under {strategy.get("max_words", 90)} words — count carefully
- Do NOT start with a salutation — go straight into the message
- Return ONLY the message text

JOB DESCRIPTION (reference):
{job_description}
""",
            },
        ],
    )
    return response.choices[0].message.content.strip()


# =============================================================================
# PHASE 7 — POLISH LINKEDIN MESSAGE
# =============================================================================

def _polish_linkedin_message(
    draft: str,
    strategy: dict,
    recruiter_analysis: dict | None = None,
) -> str:
    hook_check = ""
    if recruiter_analysis and recruiter_analysis.get("top_hook"):
        hook_check = (
            f"\n6. Verify the personalization hook ({recruiter_analysis.get('top_hook')}) "
            f"is present and reads as a warm, genuine observation — not a researched opener."
        )

    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0.4,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a ruthless editor specialized in professional outreach. "
                    "Cut every unnecessary word. "
                    "Make the message feel more human and less like a template. "
                    "If anything sounds like it was written by AI, rewrite it."
                ),
            },
            {
                "role": "user",
                "content": f"""
Polish this LinkedIn message.

STRATEGY CHECKLIST:
- Tone: {strategy.get("tone_guidance")}
- Max words: {strategy.get("max_words", 90)}
- Avoid: {json.dumps(strategy.get("what_to_avoid", []))}
- CTA should be: {strategy.get("cta")}

EDITS MUST:
1. Cut every word that doesn't earn its place
2. Make the opening line sharper and more specific
3. Ensure the value claim is backed by a concrete fact, not a vague statement
4. Make the CTA feel natural, not scripted
5. Ensure it reads like a confident human, not an AI{hook_check}

Return ONLY the polished message.

DRAFT:
{draft}
""",
            },
        ],
    )
    return response.choices[0].message.content.strip()


# =============================================================================
# PHASE 5 — ANALYSIS POLISH
# =============================================================================

def _improve_analysis(analysis: dict) -> dict:
    response = client.chat.completions.create(
        model="gpt-4.1",
        response_format={"type": "json_object"},
        temperature=0.5,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a ruthless career editor. "
                    "Upgrade content to elite level: more specific, less generic, "
                    "more impactful, confident but not arrogant. "
                    "Cut any strength that is not backed by specific evidence. "
                    "Return ONLY valid JSON with the SAME structure."
                ),
            },
            {
                "role": "user",
                "content": f"""
Improve these fields. Return the SAME JSON structure.

Focus:
- resume_bullets: tighter, more quantified, action-first, no passive voice
- strengths: remove anything generic — each must cite specific resume evidence
- gaps: honest, constructive — each is a specific, addressable delta

Do not change fit_score or key_requirements.

CONTENT:
{json.dumps(analysis, indent=2)}
""",
            },
        ],
    )
    return json.loads(response.choices[0].message.content)


# =============================================================================
# PHASE 5 — COMPANY RESEARCH (new)
# =============================================================================

def _generate_company_research(job_description: str, jd_intel: dict) -> dict:
    """
    Synthesize company intelligence from JD signals.

    Gives the candidate an insider understanding of what the company is building,
    what problem this hire solves, and how to walk into the interview with a
    strategic edge.
    """
    response = client.chat.completions.create(
        model="gpt-4.1",
        response_format={"type": "json_object"},
        temperature=0.5,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a veteran executive recruiter and talent intelligence analyst. "
                    "You read job descriptions like a detective — extracting what the company is "
                    "really trying to solve, not just what they wrote. "
                    "Be specific and opinionated. Vague observations are useless. "
                    "Return ONLY valid JSON."
                ),
            },
            {
                "role": "user",
                "content": f"""
Extract every intelligence signal from this job description.
Think like a consultant preparing a candidate for their most important interview.

EXTRACTED JD SIGNALS:
- Company: {jd_intel.get("company_name", "")}
- Stage: {jd_intel.get("company_stage", "")}
- Industry: {jd_intel.get("industry", "")}
- Culture signals: {json.dumps(jd_intel.get("company_culture_signals", []))}
- Success metrics: {json.dumps(jd_intel.get("success_metrics", []))}
- Red flags: {json.dumps(jd_intel.get("red_flags", []))}
- Value proposition: {jd_intel.get("value_proposition", "")}

Return JSON:
{{
  "company_snapshot": "string — 2–3 sentences: what this company does, its stage, and market position as inferred from the JD",
  "what_problem_this_hire_solves": "string — what specific business problem or gap is this role filling? WHY are they hiring now?",
  "what_success_looks_like_90_days": "string — based on the JD, what would make the hiring manager feel great about this hire after 3 months?",
  "culture_read": "string — what does the JD's language reveal about working style, values, and culture? Be specific.",
  "what_to_research_before_interview": [
    "specific things to Google, read, or look up before the interview"
  ],
  "smart_talking_points": [
    "specific, impressive things to mention that show deep understanding of their business"
  ],
  "questions_to_probe_in_interview": [
    "smart questions to ask that signal strategic thinking and surface important information about the role"
  ],
  "potential_red_flags_to_probe": [
    "concerns or ambiguities in the JD worth clarifying"
  ],
  "interview_angle": "string — the single most compelling narrative angle this candidate should adopt to nail the interview"
}}

JOB DESCRIPTION:
{job_description}
""",
            },
        ],
    )
    return json.loads(response.choices[0].message.content)


# =============================================================================
# PHASE 6 — TAILORED RESUME REWRITE
# =============================================================================

def _generate_tailored_resume(
    job_description: str,
    resume: str,
    analysis: dict,
    cl_strategy: dict,
    jd_intel: dict | None = None,
    resume_intel: dict | None = None,
) -> dict:
    """
    Rewrite the resume tailored to the specific role.

    Output is structured JSON rendered into a clean one-page PDF.
    Roles at the same company are grouped under one company banner.
    Content is constrained to fit one printed page.
    """
    quant_text = "\n".join(
        f"  • [{a.get('company','')}] {a.get('raw_text','')}"
        for a in (resume_intel or {}).get("quantified_achievements", [])
    ) or "  (none extracted — use only qualitative facts)"

    response = client.chat.completions.create(
        model="gpt-4.1",
        response_format={"type": "json_object"},
        temperature=0.35,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an elite resume writer and ATS optimization expert. "
                    "INTEGRITY RULE: You may ONLY use facts that appear in the original resume. "
                    "Never invent, assume, or embellish company names, titles, dates, "
                    "degrees, schools, certifications, metrics, or responsibilities. "
                    "You may reframe and reorder real facts — you may never fabricate them. "
                    "ATS RULE: The summary's first sentence must contain the exact job title and "
                    "at least 2 exact-match keywords from the must-have requirements. "
                    "Return ONLY valid JSON."
                ),
            },
            {
                "role": "user",
                "content": f"""
Rewrite this resume tailored specifically for the job below.

HARD CONSTRAINTS (non-negotiable):
- ONE printed page (0.5-inch margins). Be ruthless.
- Summary: exactly 2–3 sentences. First sentence MUST contain the exact role title and 2+ ATS keywords.
- Group all roles at the same company under ONE entry. Do not list the same company twice.
- Max 3 bullets per role. Only the most JD-relevant.
- CRITICAL: Do NOT invent companies, titles, dates, degrees, certifications, or metrics.
- Do NOT include street address.
- Omit or trim older/less-relevant roles to fit one page.

VERIFIED QUANTIFIED ACHIEVEMENTS (ONLY these numbers may appear in bullets):
{quant_text}

CANDIDATE SKILLS TAXONOMY:
{json.dumps((resume_intel or {}).get("skills", {}), indent=2)}

ATS KEYWORDS TO WEAVE IN:
{json.dumps((jd_intel or {}).get("ats_keywords", []))}

TAILORING:
- Lead bullets with strong past-tense action verbs mirroring JD vocabulary
- Quantify bullets using ONLY numbers from the verified list above
- Surface transferable skills from other industries that speak to this role
- Mirror exact JD keywords and phrases naturally

Key requirements to prioritize: {json.dumps(analysis.get("key_requirements", []))}
Language to mirror: {json.dumps(cl_strategy.get("company_language_to_mirror", []))}
Polished bullets to use/adapt: {json.dumps(analysis.get("resume_bullets", []))}

Return JSON:
{{
  "candidate_name": "string — exact from resume",
  "contact_line": "string — email · phone · LinkedIn (one line, exact from resume)",
  "summary": "string — 2–3 sentences targeted to this role",
  "experience": [
    {{
      "company": "string — exact company name from resume",
      "location": "string — City, ST",
      "roles": [
        {{
          "title": "string — exact title from resume",
          "dates": "string — e.g. March 2020 – Present",
          "bullets": ["string", "string", "string"]
        }}
      ]
    }}
  ],
  "skills": ["string"],
  "education": [{{"school": "string", "degree": "string", "dates": "string"}}],
  "certifications": ["string"]
}}

JOB DESCRIPTION:
{job_description}

ORIGINAL RESUME (source of truth):
{resume}
""",
            },
        ],
    )
    return json.loads(response.choices[0].message.content)


# =============================================================================
# PHASE 7 — RESUME OPTIMIZATION CRITIQUE
# =============================================================================

def _generate_resume_optimization_prompt(
    job_description: str,
    resume: str,
    tailored_resume: dict,
    analysis: dict,
    jd_intel: dict | None = None,
) -> dict:
    """
    Harsh-but-constructive second-reviewer critique before the final refinement pass.

    Identifies missing keywords, weak bullets, untapped transferables, language gaps,
    and hidden wins — all grounded in facts from the original resume.
    """
    response = client.chat.completions.create(
        model="gpt-4.1",
        response_format={"type": "json_object"},
        temperature=0.5,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a world-class executive career coach who has reviewed thousands of resumes. "
                    "You know exactly what gets recruiter callbacks. "
                    "Critique this tailored resume against the JD. Find every gap and missed opportunity. "
                    "Be specific — vague feedback is useless. "
                    "CRITICAL: Only suggest improvements grounded in facts from the original resume. "
                    "Never suggest inventing information. "
                    "Return ONLY valid JSON."
                ),
            },
            {
                "role": "user",
                "content": f"""
Review this tailored resume against the job description.

MUST-HAVES NOT YET MET (check coverage):
{json.dumps((jd_intel or {}).get("must_have_requirements", []), indent=2)}

ATS KEYWORDS TO CHECK COVERAGE:
{json.dumps((jd_intel or {}).get("ats_keywords", []), indent=2)}

CRITIQUE DIMENSIONS:
1. High-value JD keywords and must-haves missing or underrepresented
2. Bullets too generic — need more JD-specific language
3. Transferable skills from the original resume not surfaced
4. Summary — is it immediately compelling and ATS-optimized for this role?
5. Bullets that could be quantified (only using numbers from the original resume)
6. Vocabulary mismatches — resume uses different words for JD concepts
7. Achievements in the original resume that would resonate here but aren't highlighted
8. Anything a recruiter scanning for 6 seconds would miss

Return JSON:
{{
  "missing_keywords": ["high-value JD keywords absent from the tailored resume"],
  "weak_bullets": [
    {{
      "original": "the weak bullet",
      "issue": "why it's weak for this role",
      "improvement_direction": "how to fix it using only facts from the original resume"
    }}
  ],
  "untapped_transferable_skills": ["skill or experience from original resume not surfaced"],
  "summary_critique": "what's wrong and exactly how to improve it",
  "language_gaps": [{{"resume_uses": "string", "jd_uses": "string"}}],
  "hidden_wins": ["specific achievement from the original resume not highlighted but relevant"],
  "overall_optimization_brief": "2–3 sentences: the highest-impact changes only"
}}

JOB DESCRIPTION:
{job_description}

ORIGINAL RESUME:
{resume}

CURRENT TAILORED RESUME:
{json.dumps(tailored_resume, indent=2)}

KEY REQUIREMENTS:
{json.dumps(analysis.get("key_requirements", []), indent=2)}
""",
            },
        ],
    )
    return json.loads(response.choices[0].message.content)


# =============================================================================
# PHASE 8 — RESUME REFINEMENT
# =============================================================================

def _refine_tailored_resume(
    job_description: str,
    resume: str,
    tailored_resume: dict,
    optimization: dict,
    resume_intel: dict | None = None,
) -> dict:
    """Apply the optimization brief to produce the final maximally effective resume."""
    quant_text = "\n".join(
        f"  • [{a.get('company','')}] {a.get('raw_text','')}"
        for a in (resume_intel or {}).get("quantified_achievements", [])
    ) or "  (none extracted)"

    response = client.chat.completions.create(
        model="gpt-4.1",
        response_format={"type": "json_object"},
        temperature=0.25,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an elite resume writer applying a detailed optimization brief. "
                    "INTEGRITY RULE: Every fact in your output must exist in the original resume. "
                    "Never invent companies, titles, dates, degrees, certifications, or metrics. "
                    "Reframe and reorder real facts to maximize impact — never fabricate. "
                    "Return ONLY valid JSON."
                ),
            },
            {
                "role": "user",
                "content": f"""
Apply every suggestion from the optimization brief to produce the final resume.

OPTIMIZATION BRIEF:
{json.dumps(optimization, indent=2)}

VERIFIED QUANTIFIED ACHIEVEMENTS (ONLY these numbers may appear in bullets):
{quant_text}

WHAT TO DO:
1. Weave missing_keywords naturally into summary and bullets, grounded in real experience
2. Fix every weak bullet using its improvement_direction (original resume facts only)
3. Surface untapped_transferable_skills in skills section or bullets where supported
4. Rewrite summary per summary_critique — immediately compelling and ATS-optimized
5. Fix language_gaps — use JD vocabulary where resume used different terms
6. Surface hidden_wins if they appear in the original resume
7. North star: overall_optimization_brief
8. For any quantified bullet, use ONLY numbers from the verified achievements above

HARD CONSTRAINTS:
- ONE printed page (0.5-inch margins)
- Summary: 2–3 sentences max
- Max 3 bullets per role
- No invented facts, dates, degrees, or metrics
- Group all roles at same company under one entry
- No street address

Return SAME JSON structure, fully optimized:
{{
  "candidate_name": "string",
  "contact_line": "string",
  "summary": "string",
  "experience": [
    {{
      "company": "string",
      "location": "string",
      "roles": [{{"title": "string", "dates": "string", "bullets": ["string"]}}]
    }}
  ],
  "skills": ["string"],
  "education": [{{"school": "string", "degree": "string", "dates": "string"}}],
  "certifications": ["string"]
}}

ORIGINAL RESUME (source of truth):
{resume}

CURRENT TAILORED RESUME:
{json.dumps(tailored_resume, indent=2)}

JOB DESCRIPTION:
{job_description}
""",
            },
        ],
    )
    return json.loads(response.choices[0].message.content)


# =============================================================================
# PHASE 9 — FACT VERIFICATION
# =============================================================================

def _verify_factual_accuracy(
    cover_letter: str,
    linkedin_message: str,
    tailored_resume: dict,
    resume: str,
    resume_intel: dict | None = None,
) -> dict:
    """
    Audit all AI-generated prose against the candidate's original resume.

    Temperature 0.1 makes this behave like a deterministic fact-checker.
    Every specific claim must be traceable to the original resume.
    The pre-extracted quantified_achievements list serves as a secondary
    allowlist for numeric claims.
    """
    resume_bullets = "\n".join(
        f"  • {b}"
        for exp in (tailored_resume.get("experience") or [])
        for role in (exp.get("roles") or [])
        for b in (role.get("bullets") or [])
    )
    quant_allowlist = "\n".join(
        f"  • [{a.get('company','')}] {a.get('raw_text','')}"
        for a in (resume_intel or {}).get("quantified_achievements", [])
    ) or "  (none extracted)"

    response = client.chat.completions.create(
        model="gpt-4.1",
        response_format={"type": "json_object"},
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict factual accuracy auditor for job application materials. "
                    "Verify every specific claim against the candidate's original resume. "
                    "\n\nFACTUAL CLAIMS include: numbers, percentages, dollar amounts, dates, "
                    "company names, job titles, team sizes, technologies, degrees, schools, "
                    "certifications, product names, and any concrete verifiable detail. "
                    "\n\nVERDICT RULES:\n"
                    "- PASS: claim appears in the resume verbatim or clearly paraphrased.\n"
                    "- FAIL: claim cannot be found anywhere in the resume.\n"
                    "- When in doubt, FAIL — do not give benefit of the doubt.\n"
                    "\n\nFor FAILED claims: remove or replace with a truthful version from the resume. "
                    "Never invent a fix. Return ONLY valid JSON."
                ),
            },
            {
                "role": "user",
                "content": f"""
Audit the cover letter and LinkedIn message for factual accuracy.

ORIGINAL RESUME (only source of truth):
{resume}

VERIFIED QUANTIFIED ACHIEVEMENTS (authoritative numeric allowlist):
{quant_allowlist}

APPROVED TAILORED RESUME BULLETS (additional context):
{resume_bullets}

---
COVER LETTER TO AUDIT:
{cover_letter}

---
LINKEDIN MESSAGE TO AUDIT:
{linkedin_message}

---
Return JSON:
{{
  "cover_letter_claims": [
    {{
      "claim": "exact text",
      "resume_evidence": "quote or paraphrase from original resume, or null",
      "verdict": "PASS or FAIL",
      "action": "kept | removed | corrected"
    }}
  ],
  "linkedin_claims": [
    {{
      "claim": "exact text",
      "resume_evidence": "string or null",
      "verdict": "PASS or FAIL",
      "action": "kept | removed | corrected"
    }}
  ],
  "fabrications_removed": ["list of fabricated claims stripped out"],
  "cleaned_cover_letter": "string — cover letter with all FAIL claims removed or corrected",
  "cleaned_linkedin_message": "string — LinkedIn message with all FAIL claims removed or corrected",
  "cover_letter_accuracy_pct": 0-100,
  "linkedin_accuracy_pct": 0-100,
  "had_fabrications": true or false,
  "summary": "one sentence: claims checked, failed, overall accuracy"
}}
""",
            },
        ],
    )
    return json.loads(response.choices[0].message.content)


# =============================================================================
# PHASE 9 — INTERVIEW PREP (new)
# =============================================================================

def _generate_interview_prep(
    job_description: str,
    resume: str,
    analysis: dict,
    jd_intel: dict,
    resume_intel: dict,
) -> dict:
    """
    Generate a complete, role-specific interview guide.

    - 8 likely questions with STAR-framework answer guides using real resume evidence
    - 3 danger zones (tough questions about gaps) with reframe strategies
    - 5 power questions for the candidate to ask
    - A 30-second elevator pitch tailored to this role
    """
    quant_text = "\n".join(
        f"  • [{a.get('company','')}] {a.get('raw_text','')}"
        for a in resume_intel.get("quantified_achievements", [])
    ) or "  (none extracted)"

    response = client.chat.completions.create(
        model="gpt-4.1",
        response_format={"type": "json_object"},
        temperature=0.55,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a world-class interview coach who has prepped thousands of executives. "
                    "You predict the exact questions interviewers will ask and help candidates "
                    "craft answers that are specific, memorable, and backed by real evidence. "
                    "CRITICAL: Every suggested answer must draw from the candidate's ACTUAL experience. "
                    "Never suggest fabricating or embellishing examples. "
                    "Answers grounded in real, specific evidence beat rehearsed-sounding ones every time. "
                    "Return ONLY valid JSON."
                ),
            },
            {
                "role": "user",
                "content": f"""
Build a complete interview guide for this candidate.

ROLE: {jd_intel.get("role_title", "")} at {jd_intel.get("company_name", "")}
SENIORITY: {jd_intel.get("seniority_level", "")}

MUST-HAVE REQUIREMENTS (will drive most questions):
{json.dumps(jd_intel.get("must_have_requirements", []), indent=2)}

CULTURE SIGNALS (will drive behavioral questions):
{json.dumps(jd_intel.get("company_culture_signals", []), indent=2)}

CANDIDATE'S VERIFIED ACHIEVEMENTS (use as answer evidence):
{quant_text}

CANDIDATE GAPS (will be probed):
{json.dumps(analysis.get("gaps", []), indent=2)}

CANDIDATE STRENGTHS:
{json.dumps(analysis.get("strengths", []), indent=2)}

Return JSON:
{{
  "likely_questions": [
    {{
      "question": "string — exact question as it would be asked",
      "why_they_ask": "string — what the interviewer is testing",
      "answer_framework": {{
        "situation_setup": "string — which role/project to reference and how to frame the context",
        "key_evidence": "string — specific achievement or experience from the resume to anchor the answer",
        "talking_points": ["string", "string", "string — 3 specific points to hit"],
        "strong_close": "string — how to land the answer with impact",
        "pitfall_to_avoid": "string — what would make this answer fall flat"
      }}
    }}
  ],
  "danger_zones": [
    {{
      "question": "string — the tough question about a gap or weakness",
      "why_dangerous": "string — what the interviewer is really probing",
      "reframe_strategy": "string — how to address it honestly without killing the candidacy",
      "bridging_evidence": "string — real experience from the resume that partially addresses the gap"
    }}
  ],
  "questions_to_ask": [
    {{
      "question": "string — a sharp, impressive question to ask the interviewer",
      "why_powerful": "string — what this signals about the candidate"
    }}
  ],
  "elevator_pitch": "string — a 30-second 'tell me about yourself' answer tailored to THIS role, using the candidate's real background. Write it as prose the candidate can say out loud.",
  "preparation_checklist": [
    "string — specific things to prepare, research, or practice before the interview"
  ]
}}

JOB DESCRIPTION:
{job_description}

CANDIDATE RESUME:
{resume}
""",
            },
        ],
    )
    return json.loads(response.choices[0].message.content)


# =============================================================================
# PHASE 9 — FOLLOW-UP EMAIL (new)
# =============================================================================

def _generate_follow_up_email(
    cl_strategy: dict,
    cover_letter: str,
    company_name: str,
) -> dict:
    """
    Generate a post-application follow-up email to send ~7 days after applying.

    Adds genuine value beyond 'just checking in' — references something specific
    about the role and gives the reader a reason to respond.
    """
    response = client.chat.completions.create(
        model="gpt-4.1",
        response_format={"type": "json_object"},
        temperature=0.65,
        messages=[
            {
                "role": "system",
                "content": (
                    "You write follow-up emails that actually get responses. "
                    "The key: add new value rather than just 'checking in'. "
                    "Short (under 80 words), specific, confident — never desperate. "
                    "Return ONLY valid JSON."
                ),
            },
            {
                "role": "user",
                "content": f"""
Write a follow-up email to send 7 days after applying to {company_name}.

CONTEXT:
- Culture: {cl_strategy.get("company_culture_signals", "")}
- Candidate narrative: {cl_strategy.get("candidate_narrative_angle", "")}
- Key themes from application: {json.dumps(cl_strategy.get("key_themes", []))}
- Tone: {cl_strategy.get("tone_guidance", "")}

RULES:
- Subject line: specific, NOT "Following up on my application" or similar
- Body: under 80 words
- Must add a fresh hook or new value — NOT just "I wanted to follow up"
- Reference something specific about the role or company
- Low-friction CTA
- Confident and peer-level, not deferential
- Sign off: Max

Return JSON:
{{
  "subject_line": "string",
  "body": "string — full email body",
  "send_timing": "string — e.g. '7 days after applying, Tuesday or Wednesday morning 9–10am'",
  "personalization_note": "string — what to customize before sending"
}}

ORIGINAL COVER LETTER (context on what was submitted):
{cover_letter[:800]}
""",
            },
        ],
    )
    return json.loads(response.choices[0].message.content)


# =============================================================================
# JOB SEARCH PROFILE
# =============================================================================

def generate_job_search_profile(resume: str) -> dict:
    """Analyze a resume and return an optimized job search profile."""
    response = client.chat.completions.create(
        model="gpt-4.1",
        response_format={"type": "json_object"},
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a career advisor creating optimal job search strategies. "
                    "Be precise and data-driven. Return ONLY valid JSON."
                ),
            },
            {
                "role": "user",
                "content": f"""
Analyze this resume and return a job search profile.

Return JSON:
{{
  "target_titles": ["3–5 job title variants this candidate should target"],
  "key_skills": ["5–8 most marketable skills to highlight"],
  "experience_level": "string — seniority level",
  "inferred_location": "string — city/state if detectable, else ''",
  "search_queries": [
    "exactly 3 optimized LinkedIn keyword search strings: one specific (tight role+skill), one medium-breadth, one adjacent-role transition"
  ],
  "linkedin_keywords": ["5–8 ATS keywords for LinkedIn profile optimization"],
  "compensation_range": "string — estimated market comp range for this profile"
}}

RESUME:
{resume}
""",
            },
        ],
    )
    return json.loads(response.choices[0].message.content)


# =============================================================================
# JOB PARSING AND SCORING
# =============================================================================

def parse_and_score_jobs(resume: str, scraped_results: list[dict]) -> list[dict]:
    """
    Extract and score job listings from scraped LinkedIn results.

    Returns enriched job objects with salary (actual or estimated),
    work arrangement, specific fit reasons, and gap analysis.
    """
    parts = []
    for result in scraped_results:
        if not result.get("blocked") and result.get("text"):
            parts.append(f"--- Query: {result['query']} ---\n{result['text']}")

    if not parts:
        return []

    combined_text = "\n\n".join(parts)[:20000]

    response = client.chat.completions.create(
        model="gpt-4.1",
        response_format={"type": "json_object"},
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior talent market analyst extracting and evaluating job listings. "
                    "Be precise and honest about fit — both strengths and gaps. "
                    "\n\nSALARY RULES:\n"
                    "- If salary is explicitly stated: extract verbatim, salary_source = 'listed'\n"
                    "- If not stated: estimate based on role title, seniority, industry, company type, "
                    "  and location. Mark salary_source = 'estimated'. Think like a Zillow Zestimate — "
                    "  grounded in market signals, clearly labeled.\n"
                    "Return ONLY valid JSON."
                ),
            },
            {
                "role": "user",
                "content": f"""
Extract all job listings and score each against the candidate's resume.

SCORING:
- 8–10: Strong match — candidate clearly meets key requirements
- 6–7: Solid match — good overlap with some gaps
- 5: Stretch — could do it but significant gaps exist
- Below 5: Omit
- Sort by fit_score descending; deduplicate by title+company; max 15

Return JSON:
{{
  "jobs": [
    {{
      "title": "string",
      "company": "string",
      "location": "string",
      "posted": "string or ''",
      "fit_score": integer 5-10,
      "fit_reason": "string — one sentence summary",
      "top_fit_reasons": ["2–3 specific reasons this role suits the candidate"],
      "top_gap_reasons": ["1–2 honest gaps or concerns, or []"],
      "salary_range": "string — e.g. '$120k–$150k' or '~$135k–$160k'",
      "salary_source": "listed or estimated",
      "salary_estimate_basis": "string — rationale when estimated, else ''",
      "work_arrangement": "string — Remote | Hybrid · Xd/week | On-site | Not specified",
      "apply_url": "string — LinkedIn job URL if found, else ''"
    }}
  ]
}}

CANDIDATE RESUME:
{resume}

SEARCH RESULT TEXT:
{combined_text}
""",
            },
        ],
    )
    data = json.loads(response.choices[0].message.content)
    return data.get("jobs", [])
