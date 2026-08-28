"""The scraper factory: produce, verify, and grade per-client extension configs.

The factory's only job is the founder's directive: take a dealership link,
have the model build a scraper (as a declarative extension config — Chrome
forbids shipping code), verify it end to end in this container — including a
real-Chromium simulation of the client's own extraction engine — and surface
every decision live on the portal. It never runs recurring refreshes; those
belong to each client's machine.
"""
