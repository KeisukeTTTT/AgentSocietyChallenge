# Misinformation generation prompt
MISINFORMATION_GENERATION_PROMPT = """
You are tasked with generating misinformation tweets that would appear on X (formerly Twitter).
Please generate a realistic misinformation tweet according to the following conditions:

Theme: {theme}
Misinformation Type: {misinformation_type}

Description of Misinformation Type:
{misinformation_description}

The generated misinformation tweet must meet the following conditions:
1. Clearly exhibit the characteristics of the above misinformation type
2. Appear realistic and convincing as an actual X tweet
3. Be within 140 characters (X's character limit)
4. Include typical X elements like hashtags, mentions, or URLs
5. Include specific details or numbers to enhance credibility
6. Include emotional elements or urgency to make it more shareable

Output format:
{{
  "username": "Twitter handle of the poster",
  "display_name": "Display name of the poster",
  "title": "Main subject of the tweet",
  "content": "Body of the tweet",
  "timestamp": "YYYY-MM-DD"
}}
"""

# Rebuttal generation prompt
REBUTTAL_GENERATION_PROMPT = """
You are tasked with generating a Community Note for a misinformation tweet on X (formerly Twitter).
Please generate an effective Community Note according to the following conditions:

Misinformation Tweet:
{misinformation}

Rebuttal Type: {rebuttal_type}
Description of Rebuttal Type: {rebuttal_description}

The generated Community Note must meet the following conditions:
1. Strictly follow the logical structure of the specified rebuttal type
2. Directly address the content of the misinformation tweet
3. Be within 450 characters (Community Note limit)
4. Be fact-based and well-sourced
5. Include links to reliable sources
6. Maintain a neutral and objective tone
7. Be clear and persuasive

Output format:
{{
  "rebuttal": "Body of the Community Note",
  "sources": ["URL to source 1", "URL to source 2"]
}}
"""

# Evaluation prompt
EVALUATION_PROMPT = """
You are tasked with evaluating a Community Note on X (formerly Twitter) that addresses a potentially misleading tweet.
Please evaluate based on the following information:

Tweet:
{misinformation}

Community Note:
{rebuttal}

Your stance: {agent_type}

Your past evaluation patterns:
{past_evaluations}

Your evaluation must be one of three ratings:
- helpful: The note is accurate, well-sourced, and adds valuable context
- somewhat helpful: The note has some value but also has issues or limitations
- not helpful: The note is inaccurate, misleading, or otherwise problematic

For "helpful" ratings, select one or more reasons from:
- Cites high-quality sources
- Easy to understand
- Directly addresses the post's claim
- Provides important context
- Neutral or unbiased language
- Other

For "not helpful" ratings, select one or more reasons from:
- Sources not included or unreliable
- Sources do not support note
- Incorrect information
- Opinion or speculation
- Typos or unclear language
- Misses key points or irrelevant
- Argumentative or biased language
- Note not needed on this post
- Other

For "somewhat helpful" ratings, you may select reasons from either or both of the above lists.

Important: Maintain consistency with your past evaluation patterns and your stance.

Output format:
{{
  "rating": "helpful/somewhat helpful/not helpful",
  "reasons": ["reason1", "reason2", ...],
  "explanation": "Brief explanation for your rating and selected reasons"
}}
"""

REBUTTAL_DESCRIPTIONS = {
    "Mitigation": "The note acknowledges that 'x suppresses positive result y', but argues that the causal relationship can be mitigated by means of z.",
    "Alternative": "The note acknowledges that 'x suppresses positive result y', but argues that another factor z also suppresses y, and that reducing z should be prioritized over x.",
    "No Evidence": "The note argues that there is no evidence that 'x suppresses y'.",
    "Another True Cause": "The note argues that the true cause of suppressing y is z, not x, so addressing x will not solve the problem.",
    "Missing Mechanism 1": "The note argues that x promotes another factor z, which promotes y, so in fact x promotes y rather than suppressing it.",
    "Missing Mechanism 2": "The note argues that x suppresses another factor z, which suppresses y, so in fact x promotes y rather than suppressing it.",
    "No Need to Address": "The note argues that y is sufficient as it is, and there is no need to take action for y.",
    "Negative Effect due to y": "The note argues that y results in a negative outcome z, so y is not a positive result.",
    "Positive Effects of a Different Perspective from y 1": "The note argues that x promotes a positive result z from a different perspective than y, so x should not be eliminated.",
    "Positive Effects of a Different Perspective from y 2": "The note argues that x suppresses a negative result z from a different perspective than y, so x should not be eliminated.",
}
