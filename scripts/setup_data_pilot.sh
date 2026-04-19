#!/usr/bin/env bash
set -euo pipefail

# Setup data sources for QueryConditionedF pilot run.
# - Lexique.org CSV (for HeuristicLabeler frequency bins)
# - Stub psycholinguistic corpus (one line per species, hand-seeded)
# - Wiki-fr sample (placeholder: a short fallback if no real Wiki dump available)
# Run from repo root.

DATA_DIR="data/raw"
PSYCHO_DIR="$DATA_DIR/psycho"
LEXIQUE_URL="https://www.lexique.org/databases/Lexique383/Lexique383.tsv"
LEXIQUE_TARGET="$DATA_DIR/lexique-3.83.tsv"

mkdir -p "$DATA_DIR" "$PSYCHO_DIR"

# 1) Lexique.org TSV (~25 MB)
if [[ ! -f "$LEXIQUE_TARGET" ]]; then
    echo "[setup] downloading Lexique.org TSV..."
    curl -fsSL "$LEXIQUE_URL" -o "$LEXIQUE_TARGET" || {
        echo "[setup] WARN: Lexique download failed; HeuristicLabeler will use hash-fallback"
    }
fi

# 2) Stub psycho corpus (placeholder — replace with CHILDES-fr / Lexique-Baddeley curated sets)
for sp in phono sem lex syntax; do
    f="$PSYCHO_DIR/$sp.txt"
    if [[ ! -f "$f" ]]; then
        echo "[setup] writing stub $f (5 lines, replace with real source)"
        case "$sp" in
            phono)
                cat > "$f" << 'EOF'
La pluie tombe sur la plaine, plac et plain.
Trois petits poissons pâles sous la lune lente.
Cinq sœurs sereines chuchotent une chanson.
Les chats chassent dans la chamelade chaude.
Le rossignol roucoule à l'aube rosée.
EOF
                ;;
            sem)
                cat > "$f" << 'EOF'
Le médecin examine attentivement le patient âgé.
Les chercheurs étudient l'évolution des espèces marines.
La vérité émerge lentement de la confusion initiale.
Une décision sage exige patience et discernement.
L'histoire répète des cycles que nous oublions vite.
EOF
                ;;
            lex)
                cat > "$f" << 'EOF'
La sérendipité gouverne les découvertes vraiment fécondes.
Cet ostracisme institutionnel favorise les complaisances bourgeoises.
Les épistémologues débattent de la falsifiabilité poppérienne.
La métonymie remplace souvent une notion par sa cause.
Le palimpseste révèle ses strates à l'examen attentif.
EOF
                ;;
            syntax)
                cat > "$f" << 'EOF'
Bien que les preuves manquent, le tribunal a rendu son verdict définitif.
Les enfants que nous avons rencontrés hier au parc semblent désormais introuvables.
Si vous aviez écouté plus attentivement, vous auriez compris la subtilité du raisonnement.
La théorie selon laquelle les marchés s'autorégulent reste contestée par de nombreux économistes.
Ce que je voulais souligner, c'est l'importance que revêt cette nuance pour notre analyse.
EOF
                ;;
        esac
    fi
done

# 3) Generalist FR placeholder (replace with sampled Wiki-fr / OSCAR-fr if available)
GENERALIST="$DATA_DIR/generalist.txt"
if [[ ! -f "$GENERALIST" ]]; then
    cat > "$GENERALIST" << 'EOF'
La cuisine française traditionnelle puise ses racines dans la diversité des terroirs régionaux.
Les avancées récentes en intelligence artificielle transforment radicalement la recherche scientifique.
La conservation des espèces menacées nécessite une coopération internationale soutenue.
L'urbanisme contemporain doit concilier densité raisonnable et qualité de vie résidentielle.
Les énergies renouvelables représentent un levier essentiel de la transition écologique.
La transmission orale des contes folkloriques préserve une mémoire collective précieuse.
Les neurosciences cognitives explorent les bases biologiques de la conscience humaine.
Le commerce équitable propose une alternative aux dérives du libre échange globalisé.
Les algorithmes prédictifs influencent désormais de nombreuses décisions publiques sensibles.
La biodiversité urbaine joue un rôle clé dans la régulation thermique des villes.
EOF
fi

echo "[setup] done. Files prepared:"
ls -lh "$DATA_DIR/" "$PSYCHO_DIR/"
