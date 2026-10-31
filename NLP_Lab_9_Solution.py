"""Runnable solution for NLP Lab 9.

This script implements the four requested pipelines. The small Hindi sentiment
set and the multilingual clustering corpus below are manually curated classroom
demonstrations; replace them with a held-out, human-annotated benchmark before
making research claims. The script downloads models/data from Hugging Face on
first run and writes tables, metrics, and figures to ./nlp_lab9_outputs.

Run:
    pip install -r requirements.txt
    python NLP_Lab_9_Solution.py

The seq2seq LSTM is trained from scratch on a capped slice of OPUS-100 so the
lab is runnable on a typical laptop. Its BLEU score describes that experiment;
it is not a claim of production translation quality.
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset


SEED = 42
SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]
PAD, UNK, BOS, EOS = range(4)
NER_MODEL = "Davlan/bert-base-multilingual-cased-ner-hrl"
EMBEDDING_MODEL = "sentence-transformers/LaBSE"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize(text: str) -> list[str]:
    """Whitespace-independent Unicode tokenization for the small LSTM demo."""
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


def make_vocab(texts: Iterable[str], max_size: int = 20000) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))
    vocab = {token: index for index, token in enumerate(SPECIALS)}
    for token, _ in counts.most_common(max_size - len(vocab)):
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode(text: str, vocab: dict[str, int], max_len: int = 40) -> list[int]:
    ids = [BOS]
    ids.extend(vocab.get(token, UNK) for token in tokenize(text)[: max_len - 2])
    ids.append(EOS)
    return ids


# ---------------------------------------------------------------------------
# Q1: LaBSE alignment and English-trained zero-shot Hindi sentiment transfer
# ---------------------------------------------------------------------------

PARALLEL_PAIRS = [
    ("The weather is pleasant today.", "आज मौसम सुहावना है।"),
    ("The train arrived at the station.", "ट्रेन स्टेशन पर पहुँच गई।"),
    ("Please send me the report tomorrow.", "कृपया मुझे कल रिपोर्ट भेजें।"),
    ("The children are playing in the park.", "बच्चे पार्क में खेल रहे हैं।"),
    ("The meeting starts at nine o'clock.", "बैठक नौ बजे शुरू होती है।"),
    ("We need clean water for every village.", "हमें हर गाँव के लिए स्वच्छ पानी चाहिए।"),
]

# Labels: 1 = positive, 0 = negative. This is a small, hand-labelled demo set.
HINDI_SENTIMENT_DEMO = [
    ("यह फिल्म बहुत अच्छी और मनोरंजक है।", 1),
    ("मुझे यह किताब बेहद पसंद आई।", 1),
    ("सेवा तेज़ और कर्मचारी बहुत विनम्र थे।", 1),
    ("खाना स्वादिष्ट था और कीमत भी उचित थी।", 1),
    ("यह फोन शानदार तस्वीरें लेता है।", 1),
    ("आज का अनुभव बहुत अच्छा रहा।", 1),
    ("यह जगह साफ़ और शांत है।", 1),
    ("उत्पाद उम्मीद से बेहतर निकला।", 1),
    ("कहानी दिलचस्प थी और अभिनय शानदार था।", 1),
    ("मैं इस खरीद से खुश हूँ।", 1),
    ("यह फिल्म बहुत उबाऊ और लंबी है।", 0),
    ("मुझे यह किताब बिल्कुल पसंद नहीं आई।", 0),
    ("सेवा धीमी थी और कर्मचारी रूखे थे।", 0),
    ("खाना ठंडा और बेस्वाद था।", 0),
    ("यह फोन बार-बार बंद हो जाता है।", 0),
    ("आज का अनुभव बहुत खराब रहा।", 0),
    ("कमरा गंदा और शोरगुल वाला था।", 0),
    ("उत्पाद उम्मीद से बहुत खराब निकला।", 0),
    ("कहानी उलझी हुई थी और अभिनय कमजोर था।", 0),
    ("मैं इस खरीद से निराश हूँ।", 0),
]


def q1(output_dir: Path, encoder) -> None:
    from datasets import load_dataset
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report

    print("\n=== Q1: Cross-lingual embeddings and sentiment transfer ===")
    en_pairs = [pair[0] for pair in PARALLEL_PAIRS]
    hi_pairs = [pair[1] for pair in PARALLEL_PAIRS]
    en_vec = encoder.encode(en_pairs, normalize_embeddings=True, show_progress_bar=False)
    hi_vec = encoder.encode(hi_pairs, normalize_embeddings=True, show_progress_bar=False)
    cosines = np.sum(en_vec * hi_vec, axis=1)
    alignment = pd.DataFrame(
        {"English": en_pairs, "Hindi": hi_pairs, "Cosine similarity": cosines}
    )
    alignment.to_csv(output_dir / "q1_parallel_pair_similarity.csv", index=False)
    print("Parallel-pair cosine similarities (LaBSE):")
    print(alignment.to_string(index=False, formatters={"Cosine similarity": "{:.4f}".format}))

    # SST-2 is English-only; training the classifier only on English examples
    # makes the Hindi evaluation cross-lingual transfer rather than Hindi training.
    train_ds = load_dataset("glue", "sst2", split="train").shuffle(seed=SEED)
    val_ds = load_dataset("glue", "sst2", split="validation")
    train_ds = train_ds.select(range(min(4000, len(train_ds))))
    val_ds = val_ds.select(range(min(1000, len(val_ds))))
    x_train = encoder.encode(
        train_ds["sentence"], batch_size=64, normalize_embeddings=True,
        show_progress_bar=True,
    )
    x_val = encoder.encode(
        val_ds["sentence"], batch_size=64, normalize_embeddings=True,
        show_progress_bar=True,
    )
    y_train = np.asarray(train_ds["label"], dtype=int)
    y_val = np.asarray(val_ds["label"], dtype=int)
    classifier = LogisticRegression(max_iter=1000, random_state=SEED)
    classifier.fit(x_train, y_train)
    en_pred = classifier.predict(x_val)
    en_accuracy = accuracy_score(y_val, en_pred)

    hi_texts = [item[0] for item in HINDI_SENTIMENT_DEMO]
    hi_gold = np.asarray([item[1] for item in HINDI_SENTIMENT_DEMO], dtype=int)
    hi_x = encoder.encode(hi_texts, normalize_embeddings=True, show_progress_bar=False)
    hi_pred = classifier.predict(hi_x)
    hi_accuracy = accuracy_score(hi_gold, hi_pred)
    report = pd.DataFrame(
        {"Hindi sentence": hi_texts, "Gold label": hi_gold, "Predicted label": hi_pred}
    )
    report.to_csv(output_dir / "q1_hindi_sentiment_predictions.csv", index=False)
    print(f"English SST-2 held-out accuracy: {en_accuracy:.4f} (n={len(y_val)})")
    print(f"Hindi hand-labelled demo accuracy: {hi_accuracy:.4f} (n={len(hi_gold)})")
    print("Hindi demo classification report:")
    print(classification_report(hi_gold, hi_pred, target_names=["negative", "positive"], zero_division=0))
    print("Interpretation: LaBSE places parallel text in a shared vector space. The Hindi accuracy is a small demo score, not a representative benchmark.")


# ---------------------------------------------------------------------------
# Q2: Scratch-trained LSTM encoder-decoder for English-to-Hindi translation
# ---------------------------------------------------------------------------

class ParallelTextDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]], src_vocab, tgt_vocab, max_len=40):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        source, target = self.pairs[index]
        return encode(source, self.src_vocab, self.max_len), encode(target, self.tgt_vocab, self.max_len)


def collate_parallel(batch):
    sources, targets = zip(*batch)
    src_lengths = torch.tensor([len(x) for x in sources], dtype=torch.long)
    max_src, max_tgt = max(map(len, sources)), max(map(len, targets))
    src = torch.full((len(batch), max_src), PAD, dtype=torch.long)
    tgt = torch.full((len(batch), max_tgt), PAD, dtype=torch.long)
    for i, (s, t) in enumerate(zip(sources, targets)):
        src[i, : len(s)] = torch.tensor(s)
        tgt[i, : len(t)] = torch.tensor(t)
    return src, src_lengths, tgt


class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, src, src_lengths):
        embedded = self.embedding(src)
        packed = pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, state = self.lstm(packed)
        return state


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, decoder_input, state):
        embedded = self.embedding(decoder_input)
        output, next_state = self.lstm(embedded, state)
        return self.output(output), next_state


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim=128, hidden_dim=128):
        super().__init__()
        self.encoder = LSTMEncoder(src_vocab_size, embedding_dim, hidden_dim)
        self.decoder = LSTMDecoder(tgt_vocab_size, embedding_dim, hidden_dim)

    def forward(self, src, src_lengths, decoder_input):
        context_state = self.encoder(src, src_lengths)
        logits, _ = self.decoder(decoder_input, context_state)
        return logits


def q2(output_dir: Path, train_limit: int, epochs: int) -> None:
    from datasets import load_dataset
    from sacrebleu import corpus_bleu

    print("\n=== Q2: LSTM Seq2Seq machine translation ===")
    opus = load_dataset("Helsinki-NLP/opus-100", "en-hi")
    raw_train = opus["train"].shuffle(seed=SEED)
    raw_test = opus["test"]

    def get_pairs(split, limit):
        pairs = []
        for row in split:
            en, hi = row["translation"].get("en", "").strip(), row["translation"].get("hi", "").strip()
            if en and hi:
                pairs.append((en, hi))
            if len(pairs) >= limit:
                break
        return pairs

    train_pairs = get_pairs(raw_train, train_limit)
    test_pairs = get_pairs(raw_test, 200)
    if len(train_pairs) < 100 or not test_pairs:
        raise RuntimeError("OPUS-100 returned too few usable English-Hindi pairs.")
    src_vocab = make_vocab((en for en, _ in train_pairs))
    tgt_vocab = make_vocab((hi for _, hi in train_pairs))
    train_set = ParallelTextDataset(train_pairs, src_vocab, tgt_vocab)
    train_loader = DataLoader(
        train_set, batch_size=32, shuffle=True, collate_fn=collate_parallel,
        generator=torch.Generator().manual_seed(SEED),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2Seq(len(src_vocab), len(tgt_vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    epoch_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        weighted_loss, token_count = 0.0, 0
        for src, src_lengths, tgt in train_loader:
            src, src_lengths, tgt = src.to(device), src_lengths.to(device), tgt.to(device)
            decoder_input, gold = tgt[:, :-1], tgt[:, 1:]
            optimizer.zero_grad(set_to_none=True)
            logits = model(src, src_lengths, decoder_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            valid_tokens = int((gold != PAD).sum().item())
            weighted_loss += loss.item() * valid_tokens
            token_count += valid_tokens
        epoch_loss = weighted_loss / max(token_count, 1)
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch:02d}/{epochs}: token-weighted cross-entropy={epoch_loss:.4f}")

    tgt_inverse = {v: k for k, v in tgt_vocab.items()}

    def decode_prediction(source: str, max_steps=40) -> str:
        model.eval()
        src_ids = encode(source, src_vocab)
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_len = torch.tensor([len(src_ids)], dtype=torch.long, device=device)
        with torch.no_grad():
            state = model.encoder(src_tensor, src_len)
            token = torch.tensor([[BOS]], dtype=torch.long, device=device)
            predicted = []
            for _ in range(max_steps):
                logits, state = model.decoder(token, state)
                next_id = int(logits[:, -1].argmax(dim=-1).item())
                if next_id == EOS:
                    break
                if next_id not in (PAD, BOS):
                    predicted.append(tgt_inverse.get(next_id, "<unk>"))
                token = torch.tensor([[next_id]], dtype=torch.long, device=device)
        return " ".join(predicted)

    sample_rows = []
    hypotheses, references = [], []
    for en, hi in test_pairs:
        pred = decode_prediction(en)
        sample_rows.append({"English source": en, "Reference Hindi": hi, "Predicted Hindi": pred})
        hypotheses.append(pred)
        references.append(hi)
    sample_frame = pd.DataFrame(sample_rows)
    sample_frame.to_csv(output_dir / "q2_translation_predictions.csv", index=False)
    bleu = corpus_bleu(hypotheses, [references], tokenize="none")
    print(f"Training examples: {len(train_pairs)}; held-out examples: {len(test_pairs)}; device: {device}")
    print(f"Corpus BLEU (SacreBLEU, no tokenization): {bleu.score:.2f}")
    print("Sample predictions:")
    print(sample_frame.head(10).to_string(index=False))
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, epochs + 1), epoch_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Token-weighted training cross-entropy")
    plt.title("Q2 LSTM Seq2Seq training loss")
    plt.tight_layout()
    plt.savefig(output_dir / "q2_training_loss.png", dpi=180)
    plt.close()
    print("Architecture note: the encoder's final LSTM hidden and cell states summarize the source sequence. They initialize the decoder and serve as its context. Teacher forcing feeds the gold Hindi prefix during training; inference uses greedy predictions.")


# ---------------------------------------------------------------------------
# Q3: mBERT NER trained on 10 non-Hindi high-resource languages, evaluated on Hindi
# ---------------------------------------------------------------------------

NER_SAMPLES = {
    "English": "Sundar Pichai visited New Delhi and met officials at Google.",
    "Hindi": "सुंदर पिचाई ने नई दिल्ली का दौरा किया और गूगल के अधिकारियों से मिले।",
}


def q3(output_dir: Path) -> None:
    from transformers import AutoTokenizer, pipeline

    print("\n=== Q3: Cross-lingual mBERT NER ===")
    tokenizer = AutoTokenizer.from_pretrained(NER_MODEL)
    device = 0 if torch.cuda.is_available() else -1
    ner = pipeline(
        "token-classification", model=NER_MODEL, tokenizer=tokenizer,
        aggregation_strategy="simple", device=device,
    )
    rows = []
    token_rows = []
    for language, text in NER_SAMPLES.items():
        for entity in ner(text):
            rows.append({
                "Language": language,
                "Text": text,
                "Entity": entity.get("word", ""),
                "Label": entity.get("entity_group", entity.get("entity", "")),
                "Confidence": float(entity.get("score", 0.0)),
            })
        pieces = tokenizer.tokenize(text)
        token_rows.append({
            "Language": language,
            "Text": text,
            "WordPiece tokens": " ".join(pieces),
            "Subword count": len(pieces),
            "Unknown token count": sum(piece == tokenizer.unk_token for piece in pieces),
        })
    entity_frame = pd.DataFrame(rows, columns=["Language", "Text", "Entity", "Label", "Confidence"])
    token_frame = pd.DataFrame(token_rows)
    entity_frame.to_csv(output_dir / "q3_ner_entities.csv", index=False)
    token_frame.to_csv(output_dir / "q3_tokenization_comparison.csv", index=False)
    print(f"Model: {NER_MODEL}")
    print("Predicted entities (PER/LOC/ORG labels and confidence):")
    print(entity_frame.to_string(index=False, formatters={"Confidence": "{:.4f}".format}) if rows else "No entities emitted for these samples.")
    print("English vs Devanagari tokenization:")
    print(token_frame[["Language", "WordPiece tokens", "Subword count", "Unknown token count"]].to_string(index=False))
    print("Zero-shot scope: the selected mBERT NER checkpoint was trained on Arabic, German, English, Spanish, French, Italian, Latvian, Dutch, Portuguese, and Chinese, not Hindi. Hindi predictions are therefore cross-lingual transfer to an unseen training language. The sample is illustrative; a gold-labelled Hindi set is needed for F1.")


# ---------------------------------------------------------------------------
# Q4: balanced 3-language / 4-topic clustering in LaBSE space
# ---------------------------------------------------------------------------

CLUSTER_CORPUS = [
    ("climate", "The city planted trees to reduce heat and improve air quality.", "शहर ने गर्मी कम करने और हवा की गुणवत्ता सुधारने के लिए पेड़ लगाए।", "La ciudad plantó árboles para reducir el calor y mejorar la calidad del aire."),
    ("climate", "Solar and wind power can lower carbon emissions.", "सौर और पवन ऊर्जा कार्बन उत्सर्जन कम कर सकती हैं।", "La energía solar y eólica puede reducir las emisiones de carbono."),
    ("climate", "A long drought damaged crops across the region.", "लंबे सूखे ने पूरे क्षेत्र की फसलों को नुकसान पहुँचाया।", "Una larga sequía dañó los cultivos de toda la región."),
    ("climate", "The community restored wetlands to protect local wildlife.", "समुदाय ने स्थानीय वन्यजीवों की रक्षा के लिए आर्द्रभूमि बहाल की।", "La comunidad restauró los humedales para proteger la fauna local."),
    ("economy", "The central bank kept interest rates unchanged this month.", "केंद्रीय बैंक ने इस महीने ब्याज दरें अपरिवर्तित रखीं।", "El banco central mantuvo sin cambios las tasas de interés este mes."),
    ("economy", "Small businesses created new jobs in the town.", "छोटे व्यवसायों ने शहर में नई नौकरियाँ पैदा कीं।", "Las pequeñas empresas crearon nuevos empleos en la ciudad."),
    ("economy", "Rising food prices reduced household purchasing power.", "खाद्य पदार्थों की बढ़ती कीमतों ने परिवारों की खरीदने की क्षमता घटाई।", "El aumento de los precios de los alimentos redujo el poder adquisitivo de los hogares."),
    ("economy", "The company reported higher revenue in its annual results.", "कंपनी ने अपने वार्षिक नतीजों में अधिक राजस्व की सूचना दी।", "La empresa informó mayores ingresos en sus resultados anuales."),
    ("health", "Doctors recommend regular exercise for heart health.", "डॉक्टर हृदय के स्वास्थ्य के लिए नियमित व्यायाम की सलाह देते हैं।", "Los médicos recomiendan hacer ejercicio con regularidad para cuidar el corazón."),
    ("health", "The clinic opened a vaccination center near the school.", "क्लिनिक ने स्कूल के पास टीकाकरण केंद्र खोला।", "La clínica abrió un centro de vacunación cerca de la escuela."),
    ("health", "Researchers tested a new treatment in a clinical study.", "शोधकर्ताओं ने एक नैदानिक अध्ययन में नए उपचार का परीक्षण किया।", "Los investigadores probaron un nuevo tratamiento en un estudio clínico."),
    ("health", "Clean drinking water helps prevent many infections.", "स्वच्छ पेयजल कई संक्रमणों को रोकने में मदद करता है।", "El agua potable limpia ayuda a prevenir muchas infecciones."),
    ("technology", "The engineers designed a faster computer processor.", "इंजीनियरों ने तेज़ कंप्यूटर प्रोसेसर बनाया।", "Los ingenieros diseñaron un procesador de computadora más rápido."),
    ("technology", "The software update fixed a security vulnerability.", "सॉफ़्टवेयर अपडेट ने सुरक्षा की एक कमजोरी ठीक की।", "La actualización del software corrigió una vulnerabilidad de seguridad."),
    ("technology", "A research team trained a language model on public documents.", "एक शोध दल ने सार्वजनिक दस्तावेज़ों पर भाषा मॉडल को प्रशिक्षित किया।", "Un equipo de investigación entrenó un modelo de lenguaje con documentos públicos."),
    ("technology", "The satellite sent images to the ground station.", "उपग्रह ने ग्राउंड स्टेशन को तस्वीरें भेजीं।", "El satélite envió imágenes a la estación terrestre."),
]


def q4(output_dir: Path, encoder) -> None:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        silhouette_score,
    )

    print("\n=== Q4: Multilingual semantic clustering ===")
    records = []
    for topic, en, hi, es in CLUSTER_CORPUS:
        for language, text in (("English", en), ("Hindi", hi), ("Spanish", es)):
            records.append({"topic": topic, "language": language, "text": text})
    frame = pd.DataFrame(records)
    embeddings = encoder.encode(
        frame["text"].tolist(), normalize_embeddings=True,
        show_progress_bar=False,
    )
    kmeans = KMeans(n_clusters=4, random_state=SEED, n_init=10)
    frame["cluster"] = kmeans.fit_predict(embeddings)
    silhouette = silhouette_score(embeddings, frame["cluster"], metric="cosine")
    ari = adjusted_rand_score(frame["topic"], frame["cluster"])
    language_nmi = normalized_mutual_info_score(frame["language"], frame["cluster"])
    frame.to_csv(output_dir / "q4_document_clusters.csv", index=False)
    print(f"Documents: {len(frame)} (4 topics x 4 parallel examples x 3 languages)")
    print(f"Cosine silhouette score: {silhouette:.4f}")
    print(f"Adjusted Rand index, cluster vs known topic: {ari:.4f} (1.0 means exact topic agreement)")
    print(f"Normalized mutual information, cluster vs language: {language_nmi:.4f} (0.0 means no language association)")
    print("Cluster-by-topic table:")
    print(pd.crosstab(frame["cluster"], frame["topic"]).to_string())
    print("Cluster-by-language table:")
    print(pd.crosstab(frame["cluster"], frame["language"]).to_string())

    projected = PCA(n_components=2, random_state=SEED).fit_transform(embeddings)
    frame["PC1"], frame["PC2"] = projected[:, 0], projected[:, 1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, column, title in (
        (axes[0], "topic", "Documents colored by topic"),
        (axes[1], "language", "Documents colored by language"),
    ):
        for label in sorted(frame[column].unique()):
            subset = frame[frame[column] == label]
            ax.scatter(subset["PC1"], subset["PC2"], s=45, alpha=0.8, label=label)
        ax.set_title(title)
        ax.set_xlabel("PCA component 1")
        ax.set_ylabel("PCA component 2")
        ax.legend(fontsize=8)
    fig.suptitle("LaBSE representations of parallel English, Hindi, and Spanish text")
    fig.tight_layout()
    fig.savefig(output_dir / "q4_multilingual_pca.png", dpi=180)
    plt.close(fig)
    print("Interpretation: silhouette measures compactness/separation of the K-Means clusters. ARI measures agreement with topic labels, and language NMI detects language association. Use all three together; a high silhouette alone does not prove semantic clustering.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("nlp_lab9_outputs"))
    parser.add_argument("--q2-train-limit", type=int, default=3000)
    parser.add_argument("--q2-epochs", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed()

    from sentence_transformers import SentenceTransformer

    print(f"Loading shared multilingual embedding model: {EMBEDDING_MODEL}")
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    q1(args.output_dir, encoder)
    q2(args.output_dir, args.q2_train_limit, args.q2_epochs)
    q3(args.output_dir)
    q4(args.output_dir, encoder)
    print(f"\nFinished. Generated outputs are in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
