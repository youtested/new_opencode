# Skills block — available SKILL.md entries injected into the prompt

Template with one `{entries}` placeholder (one `<skill>` per visible
skill). The model loads a skill's full text with the skill tool only
when the task matches — this block is just the menu.

```
<available_skills>
{entries}
</available_skills>
```

Each entry:

```
<skill>
<name>{name}</name>
<description>{description}</description>
</skill>
```
