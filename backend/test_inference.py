from app.models.text_detector import predict

TEST_CASES = [
    ("short_ai_1", "As an AI language model, I don't have personal opinions, but I can provide a balanced overview of the topic."),
    ("short_ai_2", "In conclusion, the aforementioned factors collectively contribute to a comprehensive understanding of the subject matter."),
    ("long_ai_1", (
        "The rapid advancement of artificial intelligence technologies has fundamentally transformed numerous industries "
        "across the global economy. From healthcare diagnostics to financial forecasting, machine learning models now "
        "play an integral role in decision-making processes that were once exclusively human domains. This paradigm shift "
        "raises important questions about accountability, transparency, and the ethical frameworks necessary to govern "
        "increasingly autonomous systems."
    )),
    ("long_ai_2", (
        "Climate change represents one of the most pressing challenges of our time, requiring coordinated global action "
        "across multiple sectors. Renewable energy adoption, sustainable agricultural practices, and international policy "
        "cooperation are essential components of any effective mitigation strategy. Furthermore, technological innovation "
        "continues to play a pivotal role in developing solutions that balance economic growth with environmental stewardship."
    )),
    ("human_social_1", "omg just saw the funniest thing at the grocery store lol cant even"),
    ("human_social_2", "ugh mondays am i right?? need coffee asap before this meeting starts"),
    ("human_email_1", (
        "Hi team, just wanted to follow up on yesterday's discussion. I think we should push the deadline back a few days "
        "since the client hasn't sent over the final assets yet. Let me know your thoughts. Thanks, Mike"
    )),
    ("human_email_2", (
        "Hey Sarah, quick question - did you get a chance to look at the budget numbers I sent over last week? "
        "No rush, just trying to close this out before Friday. Talk soon."
    )),
    ("human_dev_1", "fixed the null pointer bug in checkout.js, turns out cart was undefined on first render, added a guard clause"),
    ("human_dev_2", "pushed hotfix to staging, db migration still pending, don't merge to main until QA signs off"),
]


def run():
    print(f"{'id':<18}{'label':<15}{'confidence':<12}{'scorer':<22}{'chars':<8}{'words':<6}")
    print("-" * 90)
    for name, text in TEST_CASES:
        result = predict(text)
        print(
            f"{name:<18}{result['label']:<15}{result['confidence_ai_generated']:<12}"
            f"{result['scorer']:<22}{result['features']['character_count']:<8}{result['features']['word_count']:<6}"
        )

    # edge cases — must not throw
    print("\n--- edge cases ---")
    for name, text in [("empty", ""), ("whitespace", "   \n\t "), ("micro", "hi")]:
        result = predict(text)
        print(f"{name:<18}{result['label']:<15}{result['confidence_ai_generated']:<12}note={result.get('note')}")


if __name__ == "__main__":
    run()
