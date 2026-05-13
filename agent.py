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
    additional_context: str = "",
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

    # ── Phase 0: parallel extraction — all inputs scanned before generation ───
    _log("📄 Extracting resume, job description, and supplementary materials…")
    fns = [
        lambda: _extract_resume(resume),
        lambda: _extract_job_description(job_description),
    ]
    has_recruiter_profile = bool(recruiter_profile and recruiter_profile.strip())
    has_additional_context = bool(additional_context and additional_context.strip())
    if has_recruiter_profile:
        fns.append(lambda: _extract_recruiter_profile(recruiter_profile))
    if has_additional_context:
        fns.append(lambda: _extract_additional_context(additional_context))

    results_p0 = _par(*fns)
    resume_intel = results_p0[0]
    jd_intel     = results_p0[1]
    idx = 2
    recruiter_intel = results_p0[idx] if has_recruiter_profile else None
    if has_recruiter_profile:
        idx += 1
    additional_context_intel = results_p0[idx] if has_additional_context else None

    # ── Phase 1: parallel fit score + analysis ────────────────────────────────
    _log("📊 Scoring fit and analyzing requirements…")
    fit_result, raw_analysis = _par(
        lambda: _compute_fit_score(job_description, resume, jd_intel, resume_intel, additional_context, additional_context_intel),
        lambda: _generate_analysis(job_description, resume, jd_intel, resume_intel, additional_context, additional_context_intel),
    )
    analysis = raw_analysis
    analysis["fit_score"]          = fit_result["fit_score"]
    analysis["fit_scoring"]        = fit_result["scoring"]
    analysis["fit_rationale"]      = fit_result.get("fit_rationale", "")
    analysis["must_have_checklist"] = fit_result.get("must_have_checklist", [])
    analysis["gaps"]               = fit_result.get("gaps", analysis.get("gaps", []))
    analysis["nice_to_have_hits"]  = fit_result.get("nice_to_have_hits", [])

    # ── Phase 2: cover letter strategy ───────────────────────────────────────
    _log("🎯 Building cover letter strategy…")
    cl_strategy  = _generate_cover_letter_strategy(job_description, resume, analysis, jd_intel, additional_context, additional_context_intel)
    company_name = cl_strategy.get("company_name", "the company")

    # ── Phase 3: parallel CL draft + recruiter analysis ──────────────────────
    _log("✍️  Drafting cover letter" + (" and analyzing recruiter profile…" if recruiter_intel else "…"))
    if recruiter_intel:
        cl_draft, recruiter_analysis = _par(
            lambda: _generate_cover_letter(job_description, resume, analysis, cl_strategy, company_name, additional_context, additional_context_intel),
            lambda: _analyze_recruiter_commonalities(recruiter_intel, resume_intel),
        )
    else:
        cl_draft = _generate_cover_letter(job_description, resume, analysis, cl_strategy, company_name, additional_context, additional_context_intel)
        recruiter_analysis = None

    # ── Phase 4: CL polish + enforce ──────────────────────────────────────────
    _log("✨ Polishing cover letter…")
    cl_polished = _polish_cover_letter(cl_draft, cl_strategy, company_name)
    cl_final    = _enforce_cover_letter_format(cl_polished, company_name)

    # ── Phase 5: parallel LI strategy + analysis polish + company research ───
    _log("🔍 Building LinkedIn strategy, polishing analysis, and researching company…")
    li_strategy, polished_analysis, company_research = _par(
        lambda: _generate_linkedin_strategy(job_description, resume, analysis, cl_strategy, recruiter_analysis, additional_context, additional_context_intel),
        lambda: _improve_analysis(analysis),
        lambda: _generate_company_research(job_description, jd_intel),
    )

    # ── Phase 6: parallel LI draft + resume draft ────────────────────────────
    _log("📝 Drafting LinkedIn message and tailoring resume…")
    li_draft, tailored_resume_draft = _par(
        lambda: _generate_linkedin_message(job_description, resume, analysis, li_strategy, recruiter_analysis, additional_context, additional_context_intel),
        lambda: _generate_tailored_resume(job_description, resume, polished_analysis, cl_strategy, jd_intel, resume_intel, additional_context, additional_context_intel),
    )

    # ── Phase 7: parallel LI polish + resume critique ────────────────────────
    _log("🔬 Polishing LinkedIn message and critiquing resume…")
    li_final, optimization = _par(
        lambda: _polish_linkedin_message(li_draft, li_strategy, recruiter_analysis),
        lambda: _generate_resume_optimization_prompt(job_description, resume, tailored_resume_draft, polished_analysis, jd_intel, additional_context, additional_context_intel),
    )

    # ── Phase 8: resume refinement ────────────────────────────────────────────
    _log("⚡ Refining resume with optimization brief…")
    tailored_resume = _refine_tailored_resume(
        job_description, resume, tailored_resume_draft, optimization, resume_intel, additional_context, additional_context_intel
    )

    # ── Phase 9: parallel fact verify + interview prep + follow-up email ──────
    _log("🛡️  Verifying accuracy, generating interview prep and follow-up email…")
    fact_check, interview_prep, follow_up_email = _par(
        lambda: _verify_factual_accuracy(cl_final, li_final, tailored_resume, resume, resume_intel, additional_context, additional_context_intel),
        lambda: _generate_interview_prep(job_description, resume, polished_analysis, jd_intel, resume_intel, additional_context, additional_context_intel),
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
        "resume_intel":             resume_intel,
        "jd_intel":                 jd_intel,
        "additional_context_intel": additional_context_intel,
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


def _extract_additional_context(text: str) -> dict:
    """
    Deep structured extraction from any supplementary candidacy materials —
    performance reviews, project write-ups, LinkedIn About, personal projects, etc.

    Facts extracted here are treated as first-class source-of-truth alongside the
    resume in all downstream generation and fact-verification steps.
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
                    "Extract only what is EXPLICITLY STATED in the provided materials. "
                    "Do NOT infer, assume, or embellish anything. "
                    "Return ONLY valid JSON."
                ),
            },
            {
                "role": "user",
                "content": f"""
Extract every verifiable fact from these additional candidacy materials.
These may include performance reviews, project summaries, LinkedIn About sections,
personal/side projects, awards, publications, or any other supporting information.

Return JSON:
{{
  "projects": [
    {{
      "name": "string — project name",
      "description": "string — what it was, your role, technologies used",
      "outcomes": ["string — measurable results, impact, or outcomes"],
      "skills_demonstrated": ["string"]
    }}
  ],
  "performance_highlights": [
    "string — verbatim or close paraphrase of standout feedback, ratings, commendations, or KPIs"
  ],
  "additional_achievements": [
    {{
      "raw_text": "string — verbatim quote of any metric, result, or accomplishment",
      "context": "string — where this came from (e.g. performance review, project doc)"
    }}
  ],
  "skills_and_competencies": ["string — any skills or competencies mentioned not already on resume"],
  "bio_or_about": "string — LinkedIn About or personal bio if present",
  "awards_and_recognition": ["string"],
  "publications_and_speaking": ["string"],
  "other_notable_facts": ["string — anything else that strengthens the candidacy"]
}}

MATERIALS:
{text}
""",
            },
        ],
    )
    return json.loads(response.choices[0].message.content)


def _fmt_additional_ctx(intel: dict | None, raw: str = "") -> str:
    """Format additional context intel into a compact prompt block. Returns '' if nothing."""
    if not intel and not raw.strip():
        return ""
    parts = []
    if intel:
        projects = intel.get("projects") or []
        if projects:
            parts.append("Projects:\n" + json.dumps(projects, indent=2))
        perf = intel.get("performance_highlights") or []
        if perf:
            parts.append("Performance highlights: " + json.dumps(perf))
        achievements = intel.get("additional_achievements") or []
        if achievements:
            parts.append("Additional verified achievements:\n" + json.dumps(achievements, indent=2))
        skills = intel.get("skills_and_competencies") or []
        if skills:
            parts.append("Additional skills/competencies: " + json.dumps(skills))
        bio = intel.get("bio_or_about", "")
        if bio:
            parts.append(f"Bio/About: {bio}")
        awards = intel.get("awards_and_recognition") or []
        if awards:
            parts.append("Awards/recognition: " + json.dumps(awards))
        other = intel.get("other_notable_facts") or []
        if other:
            parts.append("Other notable facts: " + json.dumps(other))
    if not parts and raw.strip():
        parts.append(raw[:3000])
    return "\nADDITIONAL CANDIDATE MATERIALS (treat as source-of-truth alongside resume):\n" + "\n".join(parts) if parts else ""


# =============================================================================
# PHASE 1 — FIT SCORE
# =============================================================================

def _compute_fit_score(
    job_description: str,
    resume: str,
    jd_intel: dict,
    resume_intel: dict,
    additional_context: str = "",
    additional_context_intel: dict | None = None,
) -> dict:
    """
    Rubric-based fit score across 7 weighted dimensions.

    Key design decisions:
    - Model evaluates each must-have requirement individually (checklist) before
      scoring, eliminating gestalt estimation on the most important dimension.
    - Two new dimensions capture what the original rubric missed: how well the
      candidate's actual work history maps to the role's key responsibilities
      (responsibilities_match), and whether their demonstrated impact is at the
      right scale (demonstrated_impact).
    - Nice-to-haves are surfaced separately as a non-weighted bonus signal.
    - Python recomputes the weighted average; model arithmetic is not trusted.
    - Temperature 0.1 for maximum consistency across runs.
    """
    must_haves           = jd_intel.get("must_have_requirements", [])
    nice_to_haves        = jd_intel.get("nice_to_have_requirements", [])
    key_responsibilities = jd_intel.get("key_responsibilities", [])
    success_metrics      = jd_intel.get("success_metrics", [])
    jd_tools             = jd_intel.get("required_skills", {}).get("tools_and_platforms", [])
    jd_domain            = jd_intel.get("required_skills", {}).get("domain", [])

    candidate_tools = (
        resume_intel.get("skills", {}).get("tools_and_platforms", [])
        + resume_intel.get("skills", {}).get("technical", [])
    )
    candidate_domain       = resume_intel.get("skills", {}).get("domain", [])
    candidate_methodologies = resume_intel.get("skills", {}).get("methodologies", [])

    quant_text = "\n".join(
        f"  • [{a.get('company','')}] {a.get('raw_text','')}"
        for a in resume_intel.get("quantified_achievements", [])
    ) or "  (none extracted)"

    exp_summary = []
    for exp in resume_intel.get("experience", [])[:5]:
        for role in (exp.get("roles") or [])[:2]:
            exp_summary.append(
                f"{role.get('title','')} at {exp.get('company','')} ({role.get('dates','')})"
            )

    response = client.chat.completions.create(
        model="gpt-4.1",
        response_format={"type": "json_object"},
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior technical recruiter scoring candidate–role fit. "
                    "You are calibrated, honest, and consistent — you never inflate scores. "
                    "\n\nCALIBRATION ANCHORS:\n"
                    "- 9–10: Meets ALL must-haves with direct demonstrated experience, "
                    "strong tool overlap, ideal seniority, and has done work nearly identical to this role.\n"
                    "- 7–8: Meets most must-haves, clear transferable experience, seniority aligned, "
                    "minor gaps that are easily bridged.\n"
                    "- 5–6: Relevant background but 2–3 notable gaps in hard requirements or "
                    "meaningful seniority mismatch.\n"
                    "- 3–4: Significant gaps across multiple must-haves; adjacent but not ready.\n"
                    "- 1–2: Fundamental misalignment — wrong function, level, or domain.\n"
                    "\nScore each dimension INDEPENDENTLY. Do NOT let a strong overall impression "
                    "inflate individual dimension scores. Return ONLY valid JSON."
                ),
            },
            {
                "role": "user",
                "content": f"""
Score this candidate against the role.

STEP 1 — MUST-HAVE CHECKLIST (do this first, before scoring):
For each requirement below, state whether the candidate satisfies it (true/false)
and cite specific evidence from the resume. Be strict: adjacent or implied experience
does NOT count as satisfied.

MUST-HAVE REQUIREMENTS:
{json.dumps(must_haves, indent=2)}

NICE-TO-HAVE REQUIREMENTS (for bonus signal only — do not inflate must-have scores):
{json.dumps(nice_to_haves, indent=2)}

STEP 2 — CANDIDATE PROFILE:
Experience history (most recent first):
{json.dumps(exp_summary, indent=2)}

Verified quantified achievements:
{quant_text}

Tools & technologies: {json.dumps(candidate_tools)}
Domain skills: {json.dumps(candidate_domain)}
Methodologies: {json.dumps(candidate_methodologies)}
Career arc: {resume_intel.get("career_arc", "")}
Years of experience: {resume_intel.get("total_years_experience", "unknown")}
Seniority: {resume_intel.get("seniority_level", "unknown")}
Industries: {json.dumps(resume_intel.get("industries", []))}
{_fmt_additional_ctx(additional_context_intel, additional_context)}

STEP 3 — ROLE REQUIREMENTS:
Key responsibilities (what this person actually does day-to-day):
{json.dumps(key_responsibilities, indent=2)}

What success looks like in this role:
{json.dumps(success_metrics, indent=2)}

Required tools/tech: {json.dumps(jd_tools)}
Required domain skills: {json.dumps(jd_domain)}
Expected seniority: {jd_intel.get("seniority_level", "unknown")}
Industry: {jd_intel.get("industry", "unknown")}

STEP 4 — SCORE 7 DIMENSIONS (each 1–10, independently):

1. must_have_requirements (weight 0.30)
   Score derived directly from your checklist above.
   Proportion met × 10, then adjust ±1 for depth/quality of evidence.
   Hard rule: a single unmet critical requirement caps this dimension at 7.

2. technical_skill_overlap (weight 0.20)
   What share of the JD's specific tools/technologies appear in the candidate's profile?
   Score proportionally. Partial familiarity (mentioned but not central to any role) = 0.5 credit per tool.
   If nice-to-have tools are also present, note them but do not inflate the score above what hard requirements justify.

3. responsibilities_match (weight 0.15)
   How well does the candidate's actual day-to-day work history map to the key responsibilities above?
   Have they DONE this kind of work — not just had the skills — at comparable scope and complexity?
   Transferable responsibilities from different industries count if the work is substantively similar.

4. years_and_seniority (weight 0.15)
   Does total RELEVANT experience and seniority level match role requirements?
   Penalise if under- OR significantly over-qualified (overqualified candidates often churn or underperform).
   Relevant experience = roles where the core function matches, not just industry.

5. demonstrated_impact (weight 0.10)
   Do the candidate's verified achievements show impact at the scale this role demands?
   Use the success_metrics and quantified achievements as the benchmark.
   Vague achievements without metrics score lower than specific, measurable ones.

6. leadership_and_scope (weight 0.05)
   Does demonstrated team size, budget ownership, and org influence match what this role requires?

7. domain_and_industry_fit (weight 0.05)
   Does the candidate's industry and domain background transfer to this company's context?
   Transferable domain knowledge counts even across industries.

Return JSON:
{{
  "must_have_checklist": [
    {{"requirement": "string", "satisfied": true, "evidence": "specific quote or paraphrase from resume, or null if not met"}}
  ],
  "scoring": {{
    "must_have_requirements":  {{"score": 1-10, "rationale": "1 specific sentence citing checklist results"}},
    "technical_skill_overlap": {{"score": 1-10, "rationale": "1 specific sentence"}},
    "responsibilities_match":  {{"score": 1-10, "rationale": "1 specific sentence citing role history"}},
    "years_and_seniority":     {{"score": 1-10, "rationale": "1 specific sentence"}},
    "demonstrated_impact":     {{"score": 1-10, "rationale": "1 specific sentence citing an achievement"}},
    "leadership_and_scope":    {{"score": 1-10, "rationale": "1 specific sentence"}},
    "domain_and_industry_fit": {{"score": 1-10, "rationale": "1 specific sentence"}}
  }},
  "gaps": ["plain-language description of each unmet must-have requirement"],
  "nice_to_have_hits": ["each nice-to-have the candidate satisfies, with brief evidence"],
  "fit_score": "weighted integer 1-10 — computed by you as a sanity check; Python will recompute",
  "fit_rationale": "2–3 sentences: what makes this candidate competitive for the role, and what specifically holds the score back"
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
        "must_have_requirements":  0.30,
        "technical_skill_overlap": 0.20,
        "responsibilities_match":  0.15,
        "years_and_seniority":     0.15,
        "demonstrated_impact":     0.10,
        "leadership_and_scope":    0.05,
        "domain_and_industry_fit": 0.05,
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
    additional_context: str = "",
    additional_context_intel: dict | None = None,
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
{_fmt_additional_ctx(additional_context_intel, additional_context)}

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
    additional_context: str = "",
    additional_context_intel: dict | None = None,
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
                    "The candidate writes and speaks like a real human, not a LinkedIn post. "
                    "Every piece of guidance must be concrete enough that a ghostwriter could not produce buzzword salad while following it. "
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
{_fmt_additional_ctx(additional_context_intel, additional_context)}

Return JSON:
{{
  "company_name": "string — exactly as it should appear in the greeting",
  "company_culture_signals": "string — what the JD reveals about culture, tone, values",
  "candidate_narrative_angle": "string — the single most compelling story this candidate can tell for THIS role. One sentence. Be ruthlessly specific — name the type of problem, the scale, the context.",
  "opening_hook": "string — the EXACT opening sentence of the letter body. Rules: (1) Must name a SPECIFIC outcome, client type, metric, or moment from the resume — not a category. (2) Must be something only THIS candidate could write — not interchangeable with any other applicant. (3) Must be conversational and human — reads like someone speaking, not presenting. BANNED PATTERNS: anything starting with 'With over X years', 'I am writing to express', 'Having worked in', 'As a seasoned', 'I am passionate about', 'Throughout my career'. BANNED VOCABULARY anywhere in the hook: 'architected', 'leveraged', 'scalable solutions', 'accelerated adoption', 'measurable business outcomes', 'enterprise-grade', 'best-in-class', 'synergies'. SUCCESS EXAMPLE: 'When my team prevented $11.4M in customer churn last year by re-architecting BI solutions for 900+ clients, it proved how transformative the right analytics leadership can be — especially during periods of high-stakes change.'",
  "key_themes": [
    "exactly 3 strings — each must name a specific achievement with a number or named client/project, tied directly to a challenge stated in the JD. No generic themes like 'strong communication skills' or 'proven leadership'."
  ],
  "company_language_to_mirror": [
    "4–6 specific words or phrases taken verbatim from the JD"
  ],
  "tone_guidance": "string — the EXACT tone to strike. Must be specific. Example: 'Direct and self-assured — writes like someone who has solved this problem before and knows it. No enthusiasm-signaling, no deference. Concrete verbs: built, prevented, rebuilt, led, shipped. Not: architected, leveraged, accelerated, championed.'",
  "length_and_format": "string — e.g. '3 tight paragraphs, under 260 words, no bullets'",
  "what_to_avoid": [
    "list every specific phrase, pattern, or approach that would make this letter generic or AI-sounding. Always include: 'corporate buzzwords (leveraged, architected, scalable, measurable outcomes)', 'any sentence that could appear in someone else's cover letter', 'vague claims without named evidence', 'enthusiasm-signaling phrases'"
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
    additional_context: str = "",
    additional_context_intel: dict | None = None,
) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0.75,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an elite ghostwriter for senior executives. "
                    "You write cover letters that sound like a sharp human wrote them at midnight after a good day — "
                    "confident, direct, specific, and completely free of corporate language. "
                    "You follow the strategy brief with surgical precision. "
                    "BANNED WORDS AND PHRASES (using any of these is a failure): "
                    "'leveraged', 'architected', 'scalable', 'measurable business outcomes', "
                    "'accelerated adoption', 'enterprise-grade', 'best-in-class', 'synergies', "
                    "'results-driven', 'passionate about', 'excited to', 'thrilled to', "
                    "'proven track record', 'dynamic', 'innovative', 'transformative solutions', "
                    "'stakeholder alignment', 'cross-functional collaboration', 'thought leader'. "
                    "Instead: use plain, concrete verbs. Name clients. Name projects. Use real numbers in real sentences. "
                    "Write the way the candidate speaks — like someone who has done this before and doesn't need to impress anyone. "
                    "Never begin a sentence with 'I'. Every sentence earns its place or it's cut."
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
- The opening hook must be used EXACTLY as written — do not soften, generalize, or paraphrase it
- Cover all 3 key_themes with named evidence: client names, project types, specific numbers
- Weave company_language_to_mirror naturally — never force it
- Match tone_guidance and length_and_format exactly
- Do NOT include a subject line, date, or address block
- Last two lines exactly: Thanks, / Max
- Return ONLY the cover letter text — no markdown, no asterisks, no formatting characters

CONTENT RULES (non-negotiable):
- Every paragraph must contain at least one concrete, specific detail: a client name, a project type, a number, or a named outcome
- Zero tolerance for sentences that could appear in anyone else's cover letter
- If a sentence sounds like it was generated by AI — rewrite it as plain human speech
- The letter should read like the candidate is talking to someone they respect, not performing for an ATS

JOB DESCRIPTION (for reference):
{job_description}

RESUME (for reference — only use facts from here):
{resume}
{_fmt_additional_ctx(additional_context_intel, additional_context)}
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
                    "You can smell corporate buzzwords from a mile away and you cut them on sight. "
                    "You make prose feel like a real, sharp human wrote it — specific, direct, confident. "
                    "You never soften strong opening hooks. "
                    "You never allow vague sentences that could appear in anyone else's letter. "
                    "Your standard: every sentence must be so specific that removing the candidate's name "
                    "would make it obviously wrong for any other person."
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
2. Preserve the opening hook EXACTLY — do not change a single word
3. Ensure all 3 key themes land with named, specific evidence (client name, project, number) — not vague claims
4. Hunt and destroy every buzzword: 'leveraged', 'architected', 'scalable', 'measurable outcomes', 'accelerated adoption', 'enterprise-grade', 'best-in-class', 'results-driven', 'passionate', 'excited', 'thrilled', 'transformative solutions', 'cross-functional', 'stakeholder alignment'. Replace each with a plain, concrete, human alternative.
5. Eliminate any sentence that could appear in someone else's cover letter — rewrite it with a specific detail
6. Eliminate passive voice, enthusiasm-signaling, and deference
7. Final read: does this sound like a confident human talking to someone they respect? If not, fix it.
8. Ensure there is no markdown formatting in the output: no asterisks, no backticks, no brackets used as formatting

PRESERVE EXACTLY:
- First line: Hello {company_name} Recruiting Team,
- Last two lines: Thanks, / Max

Return ONLY the final polished letter. Plain text only — no markdown.

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
    additional_context: str = "",
    additional_context_intel: dict | None = None,
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
                    "Design a writing brief for a LinkedIn message from a candidate to a hiring manager or recruiter — NOT the message itself. "
                    "Platform reality: hiring teams see 50+ messages/day. Generic = ignored. "
                    "The only messages that get responses are specific, confident, and short. "
                    "The candidate is reaching out about a real opportunity — don't obscure that. "
                    "Lead with genuine value and a direct ask, not a disguised cold pitch. "
                    "Return ONLY valid JSON."
                ),
            },
            {
                "role": "user",
                "content": f"""
Design the optimal LinkedIn outreach brief for this candidate reaching out to a hiring manager or recruiter.

COMPANY: {cl_strategy.get("company_name")}
CULTURE: {cl_strategy.get("company_culture_signals")}
CANDIDATE ANGLE: {cl_strategy.get("candidate_narrative_angle")}
{personalization_section}
Return JSON:
{{
  "platform_context": "string — format guidance (e.g. 'Short message under 90 words, direct candidate-to-recruiter tone')",
  "opening_line": "string — the exact first sentence. MUST reference something specific from the JD or the shared connection. MUST NOT be: 'I came across your role', 'I noticed you're hiring', or any generic opener. The reader must feel like you wrote this specifically for them.",
  "value_hook": "string — one sentence on the specific, concrete value this candidate brings to THIS company right now, grounded in a real achievement",
  "conversation_angle": "string — how to frame this as a confident candidate directly expressing interest, leading with value rather than desperation — honest about the context, not pretending it isn't a job inquiry",
  "cta": "string — a single, low-friction call to action (e.g. a brief call or chat about the role)",
  "tone_guidance": "string — exact tone",
  "max_words": 90,
  "what_to_avoid": ["phrases that make LinkedIn messages feel generic, desperate, or AI-generated"]
}}

JOB DESCRIPTION:
{job_description}

CANDIDATE STRENGTHS:
{json.dumps(analysis.get("strengths", []), indent=2)}
{_fmt_additional_ctx(additional_context_intel, additional_context)}
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
    additional_context: str = "",
    additional_context_intel: dict | None = None,
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
                    "You write LinkedIn messages from candidates to hiring managers and recruiters that get responses. "
                    "You follow the brief exactly. Every word is deliberate. "
                    "You never use generic openers, corporate speak, or AI-sounding phrases. "
                    "The message is honest: this is a candidate expressing direct interest in a role. "
                    "It leads with real value and a confident ask — not a disguised cold pitch. "
                    "It reads like a sharp, self-assured professional reaching out, not someone begging for a chance."
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
{_fmt_additional_ctx(additional_context_intel, additional_context)}
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
                    "You are a ruthless editor specialized in candidate outreach to hiring managers and recruiters. "
                    "Cut every unnecessary word. "
                    "Make the message feel more human and less like a template. "
                    "The candidate is reaching out about a real role — don't soften or obscure that. "
                    "If anything sounds like it was written by AI or reads as evasive about intent, rewrite it."
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
    additional_context: str = "",
    additional_context_intel: dict | None = None,
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
{_fmt_additional_ctx(additional_context_intel, additional_context)}
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
    additional_context: str = "",
    additional_context_intel: dict | None = None,
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
{_fmt_additional_ctx(additional_context_intel, additional_context)}
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
    additional_context: str = "",
    additional_context_intel: dict | None = None,
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
{_fmt_additional_ctx(additional_context_intel, additional_context)}

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
    additional_context: str = "",
    additional_context_intel: dict | None = None,
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

ORIGINAL RESUME (primary source of truth):
{resume}
{_fmt_additional_ctx(additional_context_intel, additional_context)}

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
    additional_context: str = "",
    additional_context_intel: dict | None = None,
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
                    "\n\nFor the elevator_pitch specifically: write a spoken narrative, not a résumé recitation. "
                    "It must follow a clear 5-part arc — career positioning, practitioner origin, "
                    "strategic evolution, current role with specific evidence, and a genuine company-specific close. "
                    "Tone: conversational, confident, self-aware. No HR-speak. No 'passionate about'. "
                    "Numbers embedded naturally in sentences, not listed. "
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
  "elevator_pitch": "string — A spoken 'tell me about yourself' answer. Must follow this exact 5-part arc, written as flowing prose the candidate can say out loud in ~90 seconds:\\n\\n1. POSITIONING HOOK: Open with a career arc summary and the through-line — what has stayed consistent across every role. Frame it around the value the candidate sits between (e.g. clients and technical teams, strategy and execution). One to two sentences. Do NOT open with 'I am a...' or a job title.\\n\\n2. PRACTITIONER ORIGIN: Briefly establish where they started — the hands-on, technical foundation. This builds credibility and shows they earned their strategic perspective. One sentence. Past tense.\\n\\n3. CAREER EVOLUTION: Describe the natural shift toward more strategic or higher-leverage work — framed as something that became more interesting, not as a promotion checklist. One to two sentences. Use 'shifted toward' or 'over time' language.\\n\\n4. CURRENT ROLE + EVIDENCE: Describe the current or most recent role with concrete specifics — team size, customer count, ARR, named client examples, types of problems solved. Embed numbers naturally in sentences, never as a list. Include both the client-facing and the systems/process-building dimension if applicable. Two to three sentences.\\n\\n5. COMPANY-SPECIFIC CLOSE: End with a genuine, specific reason this candidate is interested in THIS company and THIS role. Reference what the JD is actually asking for and connect it directly to what the candidate has been doing. Start with 'What draws me to this conversation is...' or equivalent. One to two sentences. Never generic.\\n\\nTONE RULES (non-negotiable):\\n- Conversational. Uses em-dashes for natural pauses. Reads like a human talking, not presenting.\\n- Confident but not boastful. Self-aware about the career arc.\\n- No HR-speak: no 'passionate about', 'leverage synergies', 'results-driven', 'team player'.\\n- Numbers woven into narrative sentences, not bulleted or listed.\\n- Does not start any sentence with 'I' more than once in a row.\\n- Do NOT recite the resume chronologically. This is a story, not a timeline.",
  "preparation_checklist": [
    "string — specific things to prepare, research, or practice before the interview"
  ]
}}

JOB DESCRIPTION:
{job_description}

CANDIDATE RESUME:
{resume}
{_fmt_additional_ctx(additional_context_intel, additional_context)}
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
                    "You write follow-up emails from candidates to hiring managers and people teams that actually get responses. "
                    "The key: add new value rather than just 'checking in'. "
                    "The candidate has applied and is following up — be honest about that context. "
                    "Short (under 80 words), specific, confident — never desperate or evasive about intent. "
                    "Return ONLY valid JSON."
                ),
            },
            {
                "role": "user",
                "content": f"""
Write a follow-up email from a candidate to a hiring manager or people team, to send 7 days after applying to {company_name}.

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
- Low-friction CTA (e.g. a quick call to discuss the role)
- Confident and direct — the candidate is genuinely interested and expressing it clearly, without being deferential or grovelling
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
