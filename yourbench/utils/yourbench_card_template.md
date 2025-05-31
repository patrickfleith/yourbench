---
{{ card_data }}
---

# {{ pretty_name }}

{% if n_subsets is defined %}
> **Total Subsets:** {{ n_subsets }}
{% endif %}

{{ footer | default("", true) }}
