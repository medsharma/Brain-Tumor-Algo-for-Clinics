#!/usr/bin/env bash
# Repo-wide overclaim sweep.
#
# Greps every text file for language the evidence in this repository does not
# support. Run it before any release, any push to main, and any time UI text
# changes. Every hit must be either cut or given a specific citation with its
# dataset and conditions attached.
#
#   bash docs/overclaim_grep.sh            # all patterns
#   bash docs/overclaim_grep.sh --ui       # app/ only, the highest-stakes text
#
# Exit code is 0 always. This is a report, not a gate. Read it.

set -uo pipefail
cd "$(dirname "$0")/.."

INCLUDE=(--include=*.md --include=*.py --include=*.ipynb --include=*.json
         --include=*.yml --include=*.yaml --include=*.txt --include=*.html
         --include=*.js --include=*.ts --include=*.tsx --include=*.css
         --include=*.toml --include=*.cfg --include=*.qml --include=*.ui)

EXCLUDE=(--exclude-dir=.git --exclude-dir=__pycache__ --exclude-dir=node_modules
         --exclude-dir=fixtures --exclude-dir=prompts)

SCOPE="."
if [ "${1:-}" = "--ui" ]; then SCOPE="app"; fi

hits=0
report() {
    local label="$1" pattern="$2"
    local out
    out=$(grep -rniE "$pattern" "${INCLUDE[@]}" "${EXCLUDE[@]}" "$SCOPE" 2>/dev/null || true)
    if [ -n "$out" ]; then
        echo "### $label"
        echo "$out"
        echo
        hits=$((hits + $(echo "$out" | wc -l)))
    fi
}

echo "=== OVERCLAIM SWEEP  scope=$SCOPE  $(date -u +%Y-%m-%dT%H:%MZ) ==="
echo

# Claims about generalisation. This project has no external validation.
report "generalisation claims" \
    '\b(generali[sz]e[sd]?|generali[sz]ation|cross-(site|scanner|institution)|transfers to|works on new)\b'

# Claims about clinical readiness. There is none.
report "clinical-readiness claims" \
    '\b(clinical[- ]grade|clinically[- ](validated|proven|ready|approved)|ready for (clinical )?(use|deployment)|deployment[- ]ready|production[- ]ready|FDA|CE[- ]mark|regulatory[- ]approv)'

# Comparative and superlative claims. No benchmark supports these.
report "superiority / SOTA claims" \
    '\b(state[- ]of[- ]the[- ]art|SOTA|outperform[a-z]*|best[- ]in[- ]class|superior to|beats|world[- ]class|cutting[- ]edge)\b'

# Absolute quality words used about the model itself.
report "unqualified quality words" \
    '\b(proven|robust(ly|ness)?|highly accurate|very accurate|extremely accurate|reliable|trustworthy|flawless|excellent performance)\b'

# The leakage-inflated number. Must appear only where it is being repudiated.
report "98.9 percent, the leakage-inflated number" \
    '98\.9'

# Accuracy quoted without a dataset. Manual review needed on every hit.
report "accuracy figures, check each has its dataset named" \
    '\b(0\.9[0-9]{1,3}|9[0-9](\.[0-9]+)?%)\b.{0,40}(accura|sensitiv|specific|AUC|F1)|((accura|sensitiv|specific|AUC|F1)[a-z]*).{0,40}\b(0\.9[0-9]{1,3}|9[0-9](\.[0-9]+)?%)'

# BRISC described as external. It is 80% the training data.
report "BRISC described as external / independent" \
    'brisc.{0,60}(external|independent|unseen|held[- ]out|new data)|((external|independent)[a-z]*).{0,60}brisc'

# Diagnosis language. This tool does not diagnose.
report "diagnosis language" \
    '\b(diagnos(is|e|es|ed|tic)|detects? (a )?(tumou?r|cancer)|confirms?|rules? out)\b'

echo "=== $hits candidate lines. Every one needs a decision: cut it, or attach"
echo "    the dataset, the n, and the conditions. ==="
