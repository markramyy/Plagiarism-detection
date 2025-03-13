## 1. Direct Plagiarism Test Cases

### Example A: Nearly Identical (High Confidence Expected)

**Original:**
```
Machine learning algorithms have revolutionized data analysis by enabling computers to identify patterns and make decisions with minimal human intervention.
```

**Plagiarized:**
```
Machine learning algorithms have revolutionized data analysis by enabling computers to recognize patterns and make decisions with minimal human intervention.
```
(Only changed "identify" to "recognize")

### Example B: Minor Rearrangement (High Confidence Expected)

**Original:**
```
The Renaissance was a period of cultural rebirth that began in Italy during the 14th century and later spread throughout Europe.
```

**Plagiarized:**
```
Beginning in Italy during the 14th century and later spreading throughout Europe, the Renaissance was a period of cultural rebirth.
```
(Same content with sentence structure rearranged)

## 2. Paraphrased Plagiarism Test Cases

### Example C: Moderate Paraphrasing (Medium-High Confidence Expected)

**Original:**
```
Global warming is primarily caused by increased levels of carbon dioxide and other greenhouse gases in the atmosphere, resulting from human activities.
```

**Plagiarized:**
```
Human activities have increased the levels of carbon dioxide and other greenhouse gases in the atmosphere, which is the main cause of global warming.
```
(Same content with restructured sentence and some word substitutions)

### Example D: Extensive Paraphrasing (Medium Confidence Expected)

**Original:**
```
Quantum computing leverages quantum mechanics principles to process information, potentially solving complex problems exponentially faster than classical computers.
```

**Plagiarized:**
```
Using principles from quantum mechanics, quantum computers can process data in ways that might solve difficult computational problems much more quickly than traditional computing systems.
```
(Heavily reworded but same core content)

## 3. Similar Topic but Different Content (Low Confidence Expected)

### Example E: Related Content

**Original:**
```
The Great Depression was a severe worldwide economic downturn that began in 1929 and lasted until the late 1930s, with devastating effects on personal income, tax revenue, and international trade.
```

**Similar but Not Plagiarized:**
```
The 2008 financial crisis was triggered by the collapse of the housing market and resulted in significant job losses, bank failures, and government bailouts across multiple countries.
```
(Similar topic of economic crises but different specific events and details)

## 4. Completely Different Content (Very Low Confidence Expected)

### Example F: Unrelated Content

**Original:**
```
Photosynthesis is the process by which green plants and some other organisms convert light energy into chemical energy that can later be released to fuel the organism's activities.
```

**Non-Plagiarized:**
```
Symphony orchestras typically include four main sections of instruments: strings, woodwinds, brass, and percussion, each contributing unique timbres to the overall sound.
```
(Entirely different topics with no content overlap)

## 5. Edge Cases

### Example G: Common Phrases but Different Context

**Original:**
```
According to recent studies, regular exercise improves cardiovascular health and reduces the risk of chronic diseases such as diabetes and hypertension.
```

**Non-Plagiarized with Common Phrases:**
```
According to recent studies, reducing sugar consumption improves dental health and reduces the risk of obesity and metabolic disorders.
```
(Contains common academic phrases like "according to recent studies" and "reduces the risk of" but in different contexts)

### Example H: Technical Content with Specialized Vocabulary

**Original:**
```
In neural networks, backpropagation calculates the gradient of the loss function with respect to each weight by the chain rule, iteratively adjusting weights to minimize error.
```

**Plagiarized Technical Content:**
```
Backpropagation in neural networks uses the chain rule to calculate the gradient of the loss function with respect to the weights, iteratively adjusting them to minimize error.
```
(Technical content with limited ways to express the same concept)
