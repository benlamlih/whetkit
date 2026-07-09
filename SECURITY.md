# Security Policy

## Reporting a vulnerability

Please **do not open a public issue** for security problems.

Report vulnerabilities privately via GitHub Security Advisories:

1. Go to https://github.com/benlamlih/whetkit/security/advisories
2. Click "Report a vulnerability" and fill in the details (affected
   command/module, reproduction steps, impact).

You should receive an acknowledgement within a few days. Please give us a
reasonable window to ship a fix before any public disclosure; you will be
credited in the advisory unless you prefer otherwise.

## Scope notes

whetkit is a local-first CLI that spawns/queries MCP servers you point it at
and sends server-provided text (tool names, descriptions, results) to LLM
providers. Reports we are especially interested in:

- prompt-injection paths that bypass the untrusted-content delimiting in the
  judge/optimizer/generator prompts
- curation-plan or overlay behavior that could execute or expose anything
  beyond the origin server's own tools
- credential handling around `${ENV_VAR}` interpolation in `server.json`

## Supported versions

Only the latest released version receives security fixes.
