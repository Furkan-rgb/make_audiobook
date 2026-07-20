"""Fixtures shared by the preparation test modules.

The preface is a real one: a citation-dense academic paragraph of the kind the
adaptation rules were tuned against, kept verbatim so the tests measure the
behaviour a reviewer would see on a real book rather than on toy strings.
"""

import re

from audiobook.preparation import (
    PreparationResult,
    PreparedBook,
    PreparedChapter,
    PreparedUnit,
    ProviderMetadata,
    SourceMetadata,
    DEFAULT_PROMPT_VERSION,
    refresh_hashes,
    sentence_spans,
)


PREFACE_SOURCE = """The psychotherapy of male homosexuals has been explored for many years. What is new
in this book is the interweaving of several strands of clinical research: the development
of male gender-identity (Abelin 1975, Greenacre 1957, Greenson 1968, Greenspan 1982,
Kohl- berg 1966, LaTorre 1979, Mahler 1955, Moberly 1983, Money and Ehrhardt 1972,
Ross 1979, Stoller 1968), histories of family dynamics (Bell and Weinberg 1978, Bieber et
al. 1962, Green 1987, Higham 1976, Money and Russo 1979, Tyson 1985), and the
techniques of psychodynamic psychotherapy of male homosexuality (Gershman 1953,
Hadden 1966, Hamilton 1939, Hatterer 1970, Horner 1989, Masters and Johnson 1979,
Nun- berg 1938, Ovesey 1969, Socarides 1978, van den Aardweg 1986, Winnicott 1965).

I would like to thank a number of people, without whose help this book could not
have been written. I want to express my appreciation to my office staff—Jennie Gohn,
Margaret Guiteras, Edith Joanis, Joan Multerer, and Cindy Anctil, and very special
appreciation to my research assistant, Jeanne Armstrong, whose many hours in the
library made this book possible."""

PREFACE_PREPARED = """The psychotherapy of male homosexuals has been explored for many years. What is new in this book is the interweaving of several strands of clinical research: the development of male gender-identity, histories of family dynamics, and the techniques of psychodynamic psychotherapy of male homosexuality.

I would like to thank a number of people, without whose help this book could not have been written. I want to express my appreciation to my office staff—Jennie Gohn, Margaret Guiteras, Edith Joanis, Joan Multerer, and Cindy Anctil, and very special appreciation to my research assistant, Jeanne Armstrong, whose many hours in the library made this book possible."""


def citation_edits(source: str) -> list[dict[str, object]]:
    """What a well-behaved model returns for the preface: three deletions.

    Built by locating the parentheticals rather than by hand so the fixture
    states the model's *intent* — remove the author-year lists — while the
    exact characters, and therefore the spacing of the result, are the
    pipeline's problem to get right.
    """

    spans = sentence_spans(source)
    edits: list[dict[str, object]] = []
    for match in re.finditer(r"\([^()]*\b\d{4}\b[^()]*\)", source):
        sentence = next(
            number
            for number, (start, end) in enumerate(spans, start=1)
            if start <= match.start() and match.end() <= end
        )
        edits.append(
            {
                "sentence": sentence,
                "category": "bibliographic_citation",
                "original": match.group(0),
                "replacement": "",
                "reason": "Visual-only sourcing is difficult to listen to.",
            }
        )
    return edits


class FakeProvider:
    def __init__(self, *, model="gemma4:31b", transform=None):
        self._metadata = ProviderMetadata(
            name="fake",
            model=model,
            prompt_version=DEFAULT_PROMPT_VERSION,
            parameters={"temperature": 0.0},
        )
        self.transform = transform
        self.calls = []
        self.availability_checks = 0
        self.closed = False

    @property
    def metadata(self):
        return self._metadata

    def check_available(self):
        self.availability_checks += 1

    def prepare(self, request):
        self.calls.append(request)
        if self.transform is not None:
            result = self.transform(request)
        else:
            result = PreparationResult()
        if result.provider_metadata is None:
            result.provider_metadata = self.metadata
        return result

    def close(self):
        self.closed = True


class FakeHTTPResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, _limit=-1):
        return self.body

    def __iter__(self):
        # urlopen responses iterate line by line; /api/pull streams NDJSON.
        return iter(self.body.splitlines(keepends=True))


def make_single_unit_book(source_text, prepared_text):
    metadata = ProviderMetadata(name="fake", model="gemma4:31b")
    unit = PreparedUnit(
        unit_id="chapter-0000-unit-0000-fixture",
        position=0,
        kind="prose",
        source_text=source_text,
        prepared_text=prepared_text,
        provider_metadata=metadata,
    )
    chapter = PreparedChapter(
        index=0,
        title="Preface",
        source_text=source_text,
        normalized_text=source_text,
        units=[unit],
    )
    book = PreparedBook(
        title="Unicode Book",
        source_metadata=SourceMetadata(extra={"language": "Français"}),
        provider_metadata=metadata,
        chapters=[chapter],
    )
    return refresh_hashes(book)
