"""Main-article extraction for the selected official Kubernetes HTML pages."""

import re
from html.parser import HTMLParser

VOID = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)


class KubernetesArticle(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.divs = 0
        self.active = False
        self.skip_tag = None
        self.skip_count = 0
        self.pre = False
        self.parts = []
        self.title = []
        self.in_title = False
        self.roots = 0

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID:
            self.handle_endtag(tag)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "title":
            self.in_title = True
        if tag == "div" and "td-content" in attrs.get("class", "").split():
            self.active = True
            self.divs = 0
            self.roots += 1
        if not self.active:
            return
        if tag == "div":
            self.divs += 1
        if self.skip_tag is not None:
            if tag == self.skip_tag:
                self.skip_count += 1
            return
        classes = attrs.get("class", "")
        if tag in {"script", "style", "nav", "button", "form"} or any(
            c in classes for c in ("feedback", "td-page-meta", "pageinfo")
        ):
            if tag not in VOID:
                self.skip_tag = tag
                self.skip_count = 1
            return
        if re.fullmatch("h[1-6]", tag):
            self.parts.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "pre":
            self.pre = True
            self.parts.append("\n\n```\n")
        elif tag in {"p", "ul", "ol", "table", "blockquote", "hr"}:
            self.parts.append("\n\n")
        elif tag == "dt":
            self.parts.append("\n\n- ")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag == "tr":
            self.parts.append("\n| ")
        elif tag == "br":
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        if not self.active:
            return
        skipping = self.skip_tag is not None
        if skipping and tag == self.skip_tag:
            self.skip_count -= 1
            if not self.skip_count:
                self.skip_tag = None
        if not skipping:
            if tag == "pre":
                self.parts.append("\n```\n\n")
                self.pre = False
            elif tag == "dt":
                self.parts.append(": ")
            elif tag == "dd":
                self.parts.append("\n\n")
            elif tag in {"td", "th"}:
                self.parts.append(" | ")
            elif tag in {"p", "ul", "ol", "table", "blockquote"} or re.fullmatch("h[1-6]", tag):
                self.parts.append("\n\n")
        if tag == "div":
            self.divs -= 1
            if self.divs == 0:
                self.active = False

    def handle_data(self, data):
        if self.in_title:
            self.title.append(data)
        if self.active and self.skip_tag is None:
            self.parts.append(data if self.pre else re.sub(r"\s+", " ", data))

    def result(self):
        if self.roots != 1 or self.active:
            raise ValueError("expected one complete Kubernetes article container")
        lines = []
        incode = False
        for line in "".join(self.parts).splitlines():
            if line.strip() == "```":
                incode = not incode
                lines.append("```")
            elif incode:
                lines.append(line)
            elif line.strip():
                lines.append(line.strip())
            elif lines and lines[-1] != "":
                lines.append("")
        text = "\n".join(lines).strip() + "\n"
        if not text.startswith("# ") or len(text) < 100:
            raise ValueError("main article missing title or content")
        return "".join(self.title).strip(), text
