def get_pair_counts(words):
    pair_count = {}
    for word in words:
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            if pair in pair_count:
                pair_count[pair] += 1
            else:
                pair_count[pair] = 1
    return pair_count

def merge_pair(pair_to_merge, words):
    """Merge the given pair in all words."""
    a, b = pair_to_merge
    merged_words = []
    merge_str = a + b  # precompute merged token

    for word in words:
        new_word = []
        i = 0
        while i < len(word):
            # check if the current pair matches
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                new_word.append(merge_str)
                i += 2  # skip next symbol since merged
            else:
                new_word.append(word[i])
                i += 1
        merged_words.append(new_word)

    return merged_words

def byte_pair_encoding(corpus, num_merges=10):
    words = [list(word) + ['</w>']  for word in corpus]
    merges = []

    for _ in range(num_merges):
        pair_counts = get_pair_counts(words)
        if not pair_counts:
            break

        # Find the most frequent pair
        best_pair = None
        max_count = 0
        for pair, count in pair_counts.items():
            if count > max_count:
                best_pair = pair
                max_count = count

        merges.append(best_pair)

        # Merge it immediately
        words = merge_pair(best_pair, words)

    tokenized_words = [' '.join(word) for word in words]
    return tokenized_words, merges

# Example usage
if __name__ == "__main__":
    corpus = ["low", "lowest", "newer", "wider"]
    tokenized_words, merges = byte_pair_encoding(corpus, num_merges=10)
    print("Final Tokenized Words:", tokenized_words)
    print("Learned Merges:", merges)