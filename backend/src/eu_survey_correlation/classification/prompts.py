"""Focused prompts for the 3-stage agentic classification pipeline."""

QUESTION_CLASSIFY_PROMPT = """\
You are classifying a Eurobarometer survey question.

Determine if this question asks for citizens' FORWARD-LOOKING OPINION or PREFERENCE \
about what the EU SHOULD do (policy preference that parliament could act on), \
versus any other type of question.

FORWARD-LOOKING OPINION (opinion_forward) — the citizen expresses what SHOULD happen:
- "Should the EU provide financial support to member states?"
- "Do you think the EU should do more to fight climate change?"
- "To what extent do you agree that the EU should have more powers to deal with crises?"
- "How important is it for the EU to invest in renewable energy?"
- "Are you in favour of a common European policy on migration?"
- "Each EU Member State should have a minimum wage for workers" (agree/disagree)

NOT FORWARD-LOOKING (not_forward) — any of these disqualify:
- References an EXISTING policy/program as already in place:
  "The European Year of Skills is taking place. Tell me if it applies to you."
  "What are your thoughts about the recovery plan?" (plan already exists)
  "The EU has taken actions in response to Russia's invasion. Do you agree with: \
Granting candidate status to Ukraine" (action already taken, asking retrospective approval)
- Factual/awareness question: "Are you aware of the EU Ecolabel?"
- Abstract sentiment (not actionable): "Do you trust EU institutions?"
- Behavioral/personal: "How often do you use public transport?"
- Consumer preference: "How important is the environment when buying products?"
- Satisfaction with current state: "How satisfied are you with democracy in the EU?"

QUESTION:
{question_text}

Respond ONLY with valid JSON (no markdown, no extra text):
{{"type": "opinion_forward" or "not_forward", "explanation": "<one sentence>"}}\
"""

VOTE_CLASSIFY_PROMPT = """\
You are classifying a European Parliament vote.

Determine if this vote is on a SUBSTANTIVE policy action (legislation, regulation, \
resolution with concrete policy measures) versus a PROCEDURAL or MINOR matter.

SUBSTANTIVE — a real policy decision citizens would care about:
- "Regulation to reduce CO2 emissions by 55% by 2030"
- "Directive on adequate minimum wages in the European Union"
- "Resolution on the situation in Ukraine"
- "Regulation on a framework for the free flow of non-personal data"

NOT SUBSTANTIVE (procedural) — administrative, procedural, or trivial:
- "Request to waive immunity of MEP X"
- "Amendment to Rule 71 of the Rules of Procedure"
- Vote on individual paragraph amendments (§ 2/1, § 2/2)
- "Motion to reject the Commission's delegated act" (procedural motion)
- "Vote on the agenda"
- "Appointment of a member of the Court of Auditors"
- Purely technical corrections or codifications

VOTE SUMMARY:
{vote_summary}

Respond ONLY with valid JSON (no markdown, no extra text):
{{"type": "substantive" or "procedural", "explanation": "<one sentence>"}}\
"""

PAIR_VALIDATE_PROMPT = """\
You are evaluating whether a European Parliament vote DIRECTLY MATCHES a Eurobarometer \
survey question — meaning parliament acted on the EXACT topic citizens were asked about.

A VALID match requires ALL of these:
1. The survey asks what SHOULD be done on a SPECIFIC topic
2. The vote ACTS ON that SAME specific topic (not just the same broad domain)
3. A citizen's survey answer (agree/disagree) maps directly to a yes/no position on this vote

VALID examples:
- Survey: "Should each EU Member State have a minimum wage for workers?"
  Vote: "Directive on adequate minimum wages in the European Union"
  → VALID (survey asks about minimum wage → parliament legislates minimum wage)

- Survey: "Promoting education in developing countries should be an EU priority"
  Vote: "Resolution on EU development cooperation to enhance access to education in \
developing countries"
  → VALID (exact same topic: education in developing countries)

INVALID — same domain but DIFFERENT specific topic:
- Survey: "Are you concerned about energy prices?"
  Vote: "Regulation on energy labelling of appliances"
  → INVALID (prices ≠ labelling)

- Survey: "Do you think the EU should do more for the environment?"
  Vote: "Ban on single-use plastics"
  → INVALID (too vague a survey question — too many inferential leaps)

INVALID — the survey is about something broader or different:
- Survey: "Should the EU have a stronger role in health?"
  Vote: "EU Digital COVID Certificate Regulation"
  → INVALID (stronger role in health ≠ COVID certificates specifically)

SURVEY QUESTION:
{question_text}

VOTE SUMMARY:
{vote_summary}

Respond ONLY with valid JSON (no markdown, no extra text):
{{"valid": true or false, "explanation": "<one sentence explaining your reasoning>"}}\
"""
