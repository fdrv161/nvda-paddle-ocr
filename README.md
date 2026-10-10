# NVDA Skill

This skill packages a practical workflow for building, debugging, and packaging
NVDA add-ons from concrete feature requests.

## What It Includes

- `SKILL.md`: the primary instructions for `/nvda`
- `references/`: workflow, troubleshooting, practical guide, advanced patterns, and a compact release checklist
- `assets/templates/`: reusable add-on project templates
- `scripts/bootstrap_addon.py`: project scaffolding script

## Why It Is Split Into Minimal And Advanced Paths

Use the minimal path when the request is just:

- one script
- one gesture
- one small settings panel
- no recurring background work
- no manager dialog or CRUD workflow

Use the advanced path when the add-on needs:

- menus and a main dialog
- popup dialogs or reminders
- background polling or recurring jobs
- external API integration
- list management, CRUD, or editor dialogs
- first-open focus handling after NVDA startup

Before handing off a non-trivial add-on, run `references/release-checklist.md`.

## Safety Rule for File Creation

By default, the skill should not create files above the `nvda/` skill
directory.

The bundled bootstrap script enforces this behavior for relative paths:

- relative `output_dir` values are resolved inside `nvda/`
- relative paths that try to escape `nvda/` are rejected
- absolute paths are allowed when an agent or user intentionally targets a
  specific repository structure

## Bootstrap Usage

Generate a minimal project inside the skill directory:

```bash
python nvda/scripts/bootstrap_addon.py generated/myAddon myAddon "My Addon"
```

Generate an advanced project skeleton with menu wiring and a reusable list
manager dialog:

```bash
python nvda/scripts/bootstrap_addon.py generated/myAddon myAddon "My Addon" --advanced
```

That command creates:

- `nvda/generated/myAddon/buildVars.py`
- `nvda/generated/myAddon/manifest.ini.tpl`
- `nvda/generated/myAddon/build_addon.py`
- `nvda/generated/myAddon/addon/...`

For advanced projects, it also creates:

- `addon/globalPlugins/<addon_id>/dialogs.py`
- an advanced `__init__.py` with settings registration, menu wiring, popup helper, and focus retry hooks

The generated project's build script writes the package to its own local
`dist/` directory:

```text
nvda/generated/myAddon/dist/
```

If a project must be created elsewhere, pass an explicit absolute path.
