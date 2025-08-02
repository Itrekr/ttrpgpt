import json

def distill_essence(text: str):
    """Extract a JSON object embedded in ``text``.

    The original implementation assumed that ``text`` always contained a JSON
    snippet and would unconditionally attempt to ``json.loads`` the slice
    between the first ``{`` and the last ``}``.  In practice the model sometimes
    returns responses without such a snippet which resulted in ``JSONDecodeError``
    being raised on an empty string.  This helper now guards against that
    situation and returns an empty dictionary when no valid JSON is found.
    """
    start = text.find("{")
    end = text.rfind("}") + 1

    # If we can't find JSON braces, avoid raising an exception
    if start == -1 or end <= 0:
        return {}

    snippet = text[start:end].strip()
    if not snippet:
        return {}

    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        # When parsing still fails, return an empty dict instead of bubbling
        # the exception up to the caller.
        return {}
