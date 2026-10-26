"""Main-article extraction must survive optional HTML tags and preserve code indentation."""

import pytest

from erag.article_extraction import KubernetesArticle


def test_optional_list_tags_do_not_truncate_article_after_example():
    parser = KubernetesArticle()
    parser.feed(
        '<title>Example</title><nav>Menu</nav><div class="td-content"><h1>Example</h1>'
        "<p>Introduction with enough explanatory content for the article validation check."
        '<ul><li>First item<li>Second item</ul><div class="highlight"><pre><code>'
        "spec:\n  replicas: 3\n    nested: true\n</code></pre></div>"
        "<h2>Later section</h2><p>This later section must survive extraction and contain its complete text.</p>"
        '<div class="feedback">Was this page helpful?</div></div><footer>Footer</footer>'
    )
    title, text = parser.result()
    assert title == "Example" and "## Later section" in text
    assert "  replicas: 3\n    nested: true" in text
    assert "Menu" not in text and "Footer" not in text and "Was this page helpful?" not in text


def test_definition_lists_tables_and_self_closing_breaks_are_preserved():
    parser = KubernetesArticle()
    parser.feed(
        '<div class="td-content"><h1>Example</h1><p>Some introductory explanatory text about the settings below.</p>'
        "<dl><dt>ClusterIP</dt><dd>Internal service.</dd><dt>NodePort</dt><dd>Static node port.</dd></dl>"
        "<table><tr><th>Setting</th><th>Value</th></tr><tr><td>Count</td><td>3</td></tr></table>"
        "<br/><h2>End section</h2><p>Complete final article paragraph with enough content.</p></div>"
    )
    _, text = parser.result()
    assert "ClusterIP: Internal service." in text and "NodePort: Static node port." in text
    assert "| Count | 3 |" in text and "## End section" in text


def test_missing_or_incomplete_article_fails_closed():
    parser = KubernetesArticle()
    parser.feed("<p>Navigation only</p>")
    with pytest.raises(ValueError):
        parser.result()
