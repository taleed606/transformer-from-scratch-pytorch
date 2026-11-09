#!/usr/bin/env bash

set -euo pipefail

# Inspect the raw Multi30K English-German data used by this project.
# The script never extracts files into the dataset directory; temporary plain-text
# copies are removed automatically when the report finishes.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RAW_DIR="$PROJECT_DIR/data/multi30k-dataset/data/task1/raw"

SPLITS=(
    "train"
    "val"
    "test_2016_flickr"
)

for command in gzip awk wc mktemp date; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "Error: required command not found: $command" >&2
        exit 1
    fi
done

if [[ ! -d "$RAW_DIR" ]]; then
    echo "Error: raw dataset directory not found: $RAW_DIR" >&2
    exit 1
fi

TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/multi30k-inspection.XXXXXX")"
trap 'rm -rf "$TEMP_DIR"' EXIT

ALL_EN="$TEMP_DIR/all.en"
ALL_DE="$TEMP_DIR/all.de"
: > "$ALL_EN"
: > "$ALL_DE"

section() {
    printf '\n%s\n' "$1"
    printf '%s\n' "$(printf '%*s' "${#1}" '' | tr ' ' '=')"
}

human_size() {
    awk -v bytes="$1" 'BEGIN {
        split("B KiB MiB GiB", unit, " ")
        value = bytes + 0
        unit_index = 1
        while (value >= 1024 && unit_index < 4) {
            value /= 1024
            unit_index++
        }
        if (unit_index == 1) {
            printf "%d %s", value, unit[unit_index]
        } else {
            printf "%.1f %s", value, unit[unit_index]
        }
    }'
}

corpus_statistics() {
    local split="$1"
    local language="$2"
    local file="$3"

    awk -v split_name="$split" -v language="$language" '
        {
            sub(/\r$/, "")
            sentences++
            sentence_length = NF
            tokens += sentence_length
            length_count[sentence_length]++

            if (sentences == 1 || sentence_length < minimum) minimum = sentence_length
            if (sentence_length > maximum) maximum = sentence_length
            if (sentence_length == 0) empty++
            if (seen_sentence[$0]++ > 0) duplicate_sentences++

            for (i = 1; i <= NF; i++) {
                if (!($i in vocabulary)) {
                    vocabulary[$i] = 1
                    types++
                }
            }
        }
        END {
            median_target = int((sentences + 1) * 0.50)
            p95_target = int((sentences * 0.95) + 0.999999)
            cumulative = 0

            for (bin_length = 0; bin_length <= maximum; bin_length++) {
                cumulative += length_count[bin_length]
                if (!median_found && cumulative >= median_target) {
                    median = bin_length
                    median_found = 1
                }
                if (!p95_found && cumulative >= p95_target) {
                    p95 = bin_length
                    p95_found = 1
                }
            }

            average = sentences ? tokens / sentences : 0
            ttr = tokens ? (100 * types / tokens) : 0

            printf "%-20s %-4s %9d %11d %9d %8.2f %5d %5d %5d %5d %7d %10d %7.2f%%\n", \
                split_name, language, sentences, tokens, types, average, minimum, median, \
                p95, maximum, empty, duplicate_sentences, ttr
        }
    ' "$file"
}

alignment_statistics() {
    local split="$1"
    local english_file="$2"
    local german_file="$3"

    awk -v split_name="$split" '
        NR == FNR {
            sub(/\r$/, "")
            english_lines[FNR] = $0
            english_lengths[FNR] = NF
            english_count = FNR
            next
        }
        {
            sub(/\r$/, "")
            german_count = FNR

            if (FNR <= english_count) {
                pairs++
                english_length = english_lengths[FNR]
                german_length = NF
                english_total += english_length
                german_total += german_length

                difference = english_length - german_length
                if (difference < 0) difference = -difference
                absolute_difference_total += difference

                if (english_length == 0 || german_length == 0) {
                    empty_pairs++
                } else {
                    larger = english_length > german_length ? english_length : german_length
                    smaller = english_length < german_length ? english_length : german_length
                    ratio = larger / smaller
                    ratio_total += ratio
                    ratio_pairs++
                    if (ratio > maximum_ratio) maximum_ratio = ratio
                    if (ratio > 2) over_double++
                }

                pair_key = english_lines[FNR] SUBSEP $0
                if (seen_pair[pair_key]++ > 0) duplicate_pairs++

                sum_x += english_length
                sum_y += german_length
                sum_x2 += english_length * english_length
                sum_y2 += german_length * german_length
                sum_xy += english_length * german_length
            }
        }
        END {
            status = english_count == german_count ? "yes" : "NO"
            average_english = pairs ? english_total / pairs : 0
            average_german = pairs ? german_total / pairs : 0
            average_difference = pairs ? absolute_difference_total / pairs : 0
            average_ratio = ratio_pairs ? ratio_total / ratio_pairs : 0

            denominator_x = pairs * sum_x2 - sum_x * sum_x
            denominator_y = pairs * sum_y2 - sum_y * sum_y
            if (denominator_x > 0 && denominator_y > 0) {
                correlation = (pairs * sum_xy - sum_x * sum_y) / \
                    sqrt(denominator_x * denominator_y)
            } else {
                correlation = 0
            }

            printf "%-20s %8d %8s %7d %7d %8.2f %8.2f %9.2f %9.2f %9.2f %7.3f %7d\n", \
                split_name, pairs, status, empty_pairs, duplicate_pairs, average_english, \
                average_german, average_difference, average_ratio, maximum_ratio, \
                correlation, over_double
        }
    ' "$english_file" "$german_file"
}

length_distribution() {
    local split="$1"
    local language="$2"
    local file="$3"

    awk -v split_name="$split" -v language="$language" '
        {
            total++
            if (NF == 0) bucket[1]++
            else if (NF <= 5) bucket[2]++
            else if (NF <= 10) bucket[3]++
            else if (NF <= 15) bucket[4]++
            else if (NF <= 20) bucket[5]++
            else if (NF <= 30) bucket[6]++
            else if (NF <= 40) bucket[7]++
            else bucket[8]++
        }
        END {
            printf "%-20s %-4s", split_name, language
            for (i = 1; i <= 8; i++) {
                percentage = total ? 100 * bucket[i] / total : 0
                value = sprintf("%d (%.1f%%)", bucket[i], percentage)
                printf " %14s", value
            }
            printf "\n"
        }
    ' "$file"
}

longest_sentences() {
    local language="$1"
    local file="$2"

    awk -v language="$language" '
        {
            candidate_length = NF
            candidate_line = NR
            candidate_text = $0

            for (position = 1; position <= 5; position++) {
                if (candidate_length > top_length[position]) {
                    for (shift = 5; shift > position; shift--) {
                        top_length[shift] = top_length[shift - 1]
                        top_line[shift] = top_line[shift - 1]
                        top_text[shift] = top_text[shift - 1]
                    }
                    top_length[position] = candidate_length
                    top_line[position] = candidate_line
                    top_text[position] = candidate_text
                    break
                }
            }
        }
        END {
            for (position = 1; position <= 5; position++) {
                if (top_line[position]) {
                    text = top_text[position]
                    if (length(text) > 150) text = substr(text, 1, 147) "..."
                    printf "%-4s %4d %7d  %s\n", language, top_length[position], \
                        top_line[position], text
                }
            }
        }
    ' "$file"
}

printf 'MULTI30K RAW DATA INSPECTION REPORT\n'
printf '==================================\n'
printf 'Generated: %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"
printf 'Dataset:   Multi30K Task 1 parallel text\n'
printf 'Task:      English-to-German machine translation\n'
printf 'Raw path:  %s\n' "$RAW_DIR"
printf 'Splits:    train, val, test_2016_flickr\n'
printf 'Counting:  whitespace-separated raw tokens; capitalization preserved\n'

section "1. ACTIVE FILE INVENTORY"
printf '%-20s %12s %12s %11s %11s %10s\n' \
    "Split" "EN size" "DE size" "EN lines" "DE lines" "Aligned"
printf '%-20s %12s %12s %11s %11s %10s\n' \
    "--------------------" "------------" "------------" "-----------" "-----------" "----------"

for split in "${SPLITS[@]}"; do
    english_gzip="$RAW_DIR/$split.en.gz"
    german_gzip="$RAW_DIR/$split.de.gz"

    for file in "$english_gzip" "$german_gzip"; do
        if [[ ! -f "$file" ]]; then
            echo "Error: expected project file not found: $file" >&2
            exit 1
        fi
        gzip -t "$file"
    done

    english_plain="$TEMP_DIR/$split.en"
    german_plain="$TEMP_DIR/$split.de"
    gzip -cd "$english_gzip" > "$english_plain"
    gzip -cd "$german_gzip" > "$german_plain"

    cat "$english_plain" >> "$ALL_EN"
    cat "$german_plain" >> "$ALL_DE"

    english_bytes="$(wc -c < "$english_gzip" | tr -d ' ')"
    german_bytes="$(wc -c < "$german_gzip" | tr -d ' ')"
    english_lines="$(wc -l < "$english_plain" | tr -d ' ')"
    german_lines="$(wc -l < "$german_plain" | tr -d ' ')"

    if [[ "$english_lines" -eq "$german_lines" ]]; then
        aligned="yes"
    else
        aligned="NO"
    fi

    printf '%-20s %12s %12s %11d %11d %10s\n' \
        "$split" "$(human_size "$english_bytes")" "$(human_size "$german_bytes")" \
        "$english_lines" "$german_lines" "$aligned"
done

section "2. EXCLUDED RAW FILES"
printf 'Excluded languages: Czech (.cs.gz) and French (.fr.gz)\n'
printf 'Reason:             the project implements English-to-German translation.\n'
printf 'Additional tests:   test_2017_flickr, test_2017_mscoco, test_2018_flickr\n'
printf 'Reason:             the current preprocessing pipeline uses test_2016_flickr.\n'

section "3. CORPUS STATISTICS"
printf '%-20s %-4s %9s %11s %9s %8s %5s %5s %5s %5s %7s %10s %8s\n' \
    "Split" "Lang" "Sentences" "Tokens" "Types" "Avg len" "Min" "P50" "P95" "Max" \
    "Empty" "Duplicates" "TTR"
printf '%-20s %-4s %9s %11s %9s %8s %5s %5s %5s %5s %7s %10s %8s\n' \
    "--------------------" "----" "---------" "-----------" "---------" "--------" "-----" \
    "-----" "-----" "-----" "-------" "----------" "--------"

for split in "${SPLITS[@]}"; do
    corpus_statistics "$split" "EN" "$TEMP_DIR/$split.en"
    corpus_statistics "$split" "DE" "$TEMP_DIR/$split.de"
done
corpus_statistics "ALL_ACTIVE_SPLITS" "EN" "$ALL_EN"
corpus_statistics "ALL_ACTIVE_SPLITS" "DE" "$ALL_DE"

section "4. PARALLEL-PAIR AND LENGTH-ALIGNMENT STATISTICS"
printf '%-20s %8s %8s %7s %7s %8s %8s %9s %9s %9s %7s %7s\n' \
    "Split" "Pairs" "Aligned" "Empty" "Dup" "Avg EN" "Avg DE" "Avg |diff|" \
    "Avg ratio" "Max ratio" "Corr" ">2x"
printf '%-20s %8s %8s %7s %7s %8s %8s %9s %9s %9s %7s %7s\n' \
    "--------------------" "--------" "--------" "-------" "-------" "--------" "--------" \
    "---------" "---------" "---------" "-------" "-------"

for split in "${SPLITS[@]}"; do
    alignment_statistics "$split" "$TEMP_DIR/$split.en" "$TEMP_DIR/$split.de"
done
alignment_statistics "ALL_ACTIVE_SPLITS" "$ALL_EN" "$ALL_DE"

section "5. SENTENCE-LENGTH DISTRIBUTION"
printf '%-20s %-4s %14s %14s %14s %14s %14s %14s %14s %14s\n' \
    "Split" "Lang" "0" "1-5" "6-10" "11-15" "16-20" "21-30" "31-40" ">40"
printf '%-20s %-4s %14s %14s %14s %14s %14s %14s %14s %14s\n' \
    "--------------------" "----" "--------------" "--------------" "--------------" "--------------" \
    "--------------" "--------------" "--------------" "--------------"

for split in "${SPLITS[@]}"; do
    length_distribution "$split" "EN" "$TEMP_DIR/$split.en"
    length_distribution "$split" "DE" "$TEMP_DIR/$split.de"
done

section "6. TRAINING VOCABULARY OVERLAP"
awk '
    NR == FNR {
        for (i = 1; i <= NF; i++) {
            if (!($i in english)) {
                english[$i] = 1
                english_types++
            }
        }
        next
    }
    {
        for (i = 1; i <= NF; i++) {
            if (!($i in german)) {
                german[$i] = 1
                german_types++
                if ($i in english) overlap++
            }
        }
    }
    END {
        union_size = english_types + german_types - overlap
        printf "English types:                  %d\n", english_types
        printf "German types:                   %d\n", german_types
        printf "Exact surface-form overlap:     %d\n", overlap
        printf "Combined vocabulary (union):    %d\n", union_size
        printf "Overlap as %% of English types:  %.2f%%\n", 100 * overlap / english_types
        printf "Overlap as %% of German types:   %.2f%%\n", 100 * overlap / german_types
    }
' "$TEMP_DIR/train.en" "$TEMP_DIR/train.de"

section "7. FIVE LONGEST TRAINING SENTENCES"
printf '%-4s %4s %7s  %s\n' "Lang" "Len" "Line" "Sentence"
printf '%-4s %4s %7s  %s\n' "----" "----" "-------" "--------"
longest_sentences "EN" "$TEMP_DIR/train.en"
longest_sentences "DE" "$TEMP_DIR/train.de"

section "8. FIRST THREE TRAINING PAIRS"
awk '
    NR == FNR {
        if (FNR <= 3) english[FNR] = $0
        next
    }
    FNR <= 3 {
        printf "Pair %d\n", FNR
        printf "  EN: %s\n", english[FNR]
        printf "  DE: %s\n", $0
    }
' "$TEMP_DIR/train.en" "$TEMP_DIR/train.de"

section "9. READING THE REPORT"
printf '%s\n' \
    '- One compressed file line represents one sentence.' \
    '- Matching English/German line numbers form a translation pair.' \
    '- Length means whitespace-separated words before tokenization or BPE.' \
    '- Types are case-sensitive unique surface forms; TTR is types/tokens.' \
    '- Ratio is longer-sentence length divided by shorter-sentence length.' \
    '- Corr is the Pearson correlation between paired EN and DE lengths.' \
    '- Duplicates count repeated occurrences after the first occurrence.'
