def find_nonoverlapping_common_substrings(s1, s2, s3, min_words=5):
    norm_s1 = " ".join(s1.split())
    norm_s2 = " ".join(s2.split())
    norm_s3 = " ".join(s3.split())

    # Work with the words from the first string to extract candidates.
    words1 = norm_s1.split()
    results = []
    i = 0
    
    # Loop through each word position in s1 as a potential start of a common substring.
    while i < len(words1):
        longest = ""
        longest_length = 0
        # Try extending the candidate substring from position i.
        for j in range(i, len(words1)):
            candidate = " ".join(words1[i:j+1])
            # Check if candidate exists in both s2 and s3.
            if candidate in norm_s2 and candidate in norm_s3:
                # Only record candidates that meet the minimum words length.
                if (j - i + 1) >= min_words:
                    longest = candidate
                    longest_length = j - i + 1
            else:
                # As soon as a candidate fails, we break the inner loop.
                break
        # If a common candidate (of at least min_words) was found, add it and skip its words.
        if longest:
            results.append(longest)
            i += longest_length  # move past the found common substring
        else:
            i += 1  # otherwise, check the next starting word
            
    return results

if __name__ == "__main__":
    S1 = "In the early 1900s, amidst the shifting tides of change brought on by the Japanese annexation of Korea, a diligent fisherman named Joon and his wife Sunja found themselves navigating the nuances of survival in a coastal village. They turned their modest home into a haven for lodgers, seeking the extra income that could support their small family. Their intentions were practical rather than ambitious, driven by the need to provide a stable environment for their son, Hoonie, who had entered the world with physical differences that drew unwanted attention in their traditional community. Yet, his mind was a wellspring of intelligence, displaying an early acumen that both surprised and delighted his parents. As Japan's influence seeped deeper into Korean life, Joon and Sunja worked tirelessly, managing their lodgers with a quiet grace that belied the underlying worries of an uncertain future. They witnessed the subtle shifts in societal norms and the underlying tension that brewed under the surface of everyday life. Hoonie, observant beyond his years, absorbed the quiet resilience of his parents, understanding that their labor was both a means of survival and a shield against the vulnerabilities that beset them. He grew up learning to navigate not only his own physical limitations but also a world that was rapidly transforming around him. The village, with its changing rhythms and subdued whispers of discontent, provided a backdrop to Hoonie's quiet determination to carve out a space where he was both seen and valued. Joon, engrossed in weaving nets and measuring tides, and Sunja, with her steady hands managing the daily influx and departure of lodgers, taught him lessons of endurance without words, showing him how to find strength in simplicity and purpose in persistence. Hoonie’s coming of age was marked less by the grand events of history and more by the intimate moments of familial bonds formed within the humble walls of their home, where love, labor, and learning coalesced with the subtle resolve to endure."
    S2 = "In the early 1900s, during the Japanese annexation of Korea, a diligent fisherman named Joon and his wife Sunja lived in a coastal village. They turned their modest home into a haven for lodgers, seeking the extra income that could support their small family. Their son, Hoonie, had entered the world with physical disabilities that drew unwanted attention in their traditional community, even though his mind was a wellspring of intelligence, and his early acumen delighted his parents. As Japan's influence seeped deeper into Korean life, Joon and Sunja worked tirelessly to manage their lodgers with a quiet grace that belied the underlying worries of an uncertain future. They witnessed the underlying tension and societal shifts that disturbed everyday life. Hoonie, observant beyond his years, absorbed the resilience of his parents, understanding that their labor was a shield against vulnerabilities. He grew up learning to navigate not only his own physical limitations but also a world that was rapidly transforming around him. The village, with its discontented whispers and changing rhythms, provided a backdrop to Hoonie's determination to carve out a space where he was both seen and valued. Joon, engrossed in weaving nets and measuring tides, and Sunja, with her steady hands managing the daily influx and departure of lodgers, taught him lessons of endurance. He learned their quiet, purposeful strength. Hoonie's coming of age was marked less by the grand events of history and more by his evolving familial bonds. His home taught him the love, labor, and learning required to endure."
    S3 = "In the early 1900s, after Japan annexed Korea, a diligent fisherman named Joon and his wife Sunja began to work tirelessly to survive and protect their family in a rapidly changing coastal village. They turned their modest home into a haven for lodgers, seeking the extra income that could support their small family. They needed the extra income to take care of their son, Hoonie, a spirited child with physical disabilities. However, he had a brilliant mind, proving himself a quick learner and bringing delight to his parents. As Japan's influence seeped deeper into Korean life, Joon and Sunja worked tirelessly, managing their lodgers but fretting constantly about the uncertain future. As they worked, they witnessed how societal norms and tensions evolved in even the most mundane moments. Hoonie, observant beyond his years, absorbed the quiet resilience of his parents. He understood that their labor was both a way to make money and a way to protect him from his vulnerabilities. He grew up learning to navigate not only his own physical limitations but also a world that was rapidly transforming around him. Surrounded by a slowly modernizing village, Hoonie slowly learned how to build a space that was both seen and valued. Joon taught him how to weave nets and measure tides, while Sunja showed him how to maintain and clean a home, and host people in a way that made them feel welcome. From them, he learned about endurance, simplicity, and finding purpose in his daily tasks. Together, the family endured, keeping their small home a place of love, labor, and support."

    # Test with the example strings
    result = find_nonoverlapping_common_substrings(S1, S2, S3)
    for i, part in enumerate(result, 1):
        print(f"Common part {i}: '{part}'")
        