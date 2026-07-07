# Releasing whetkit

Releases are fully automated: **push a version tag and CI does the rest**
(test → build → publish to PyPI via Trusted Publishing → GitHub Release with
artifacts). No API tokens are stored anywhere.

## One-time setup (before the first release)

1. **Repo name**: make sure the GitHub repo is named `benlamlih/whetkit`
   (Trusted Publishing matches on the exact owner/repo).
2. **PyPI account**: create one at https://pypi.org with 2FA enabled.
3. **Register the trusted publisher**: PyPI → *Your account → Publishing →
   Add a new pending publisher* with **exactly**:
   - PyPI project name: `whetkit`
   - Owner: `benlamlih`
   - Repository name: `whetkit`
   - Workflow name: `release.yml`
   - Environment name: `pypi`

   ("Pending" because the project doesn't exist on PyPI yet — the first
   successful publish claims the name and converts it to a normal trusted
   publisher.)
4. (Recommended) GitHub → repo *Settings → Environments → `pypi`* → add
   yourself as a required reviewer. Publishing then needs a one-click
   approval, which prevents a bad tag from shipping unreviewed.

## Every release

1. Bump the version and update the pins record:

   ```sh
   uv version 0.2.0        # updates pyproject.toml + uv.lock
   # re-verify dependency pins per VERSIONS.md, update check dates
   ```

2. Commit and push (Conventional Commits):

   ```sh
   git commit -am "chore(release): v0.2.0"
   git push
   ```

3. Tag and push the tag — this is the trigger:

   ```sh
   git tag v0.2.0
   git push origin v0.2.0
   ```

4. Watch the *Release* workflow. It will:
   - fail fast if the tag doesn't match `pyproject.toml`'s version,
   - run the full test suite,
   - build the sdist + wheel with `uv build`,
   - publish to PyPI through OIDC (environment `pypi`),
   - create a GitHub Release with generated notes and the dist files.

5. Verify the release:

   ```sh
   uvx whetkit@latest --help
   ```

If the publish step fails (e.g. the trusted publisher wasn't registered
yet), fix the PyPI side and re-run the workflow from the Actions tab — the
tag doesn't need to be recreated.

## Users install with

```sh
uv tool install whetkit   # or: uvx whetkit / pipx install whetkit
```
