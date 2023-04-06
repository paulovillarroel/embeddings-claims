library(tidyverse)
library(readxl)
library(httr)


claims <- read_xlsx("raw-data/reclamos.xlsx") |>
  mutate(
    description = paste(DESCRIPCION_CIUDADANO, PETICION_CIUDADANO),
    class = factor(MERCADO_ANALISTA)
  ) |>
  select(CASO_ID, description, class) |>
  arrange(CASO_ID)


# Text embeddings

embeddings_url <- "https://api.openai.com/v1/embeddings"
auth <- add_headers(Authorization = paste("Bearer", Sys.getenv("OPENAI_API_KEY")))
body <- list(model = "text-embedding-ada-002", input = claims$description)

resp <- POST(
  embeddings_url,
  auth,
  body = body,
  encode = "json"
)

embeddings <- content(resp, as = "text", encoding = "UTF-8") |>
  jsonlite::fromJSON(flatten = TRUE) |>
  pluck("data", "embedding")

claims_embeddings <- claims |>
  mutate(embeddings = embeddings)

claims_embeddings |>
  select(CASO_ID, description, embeddings)

embeddings_mat <- matrix(
  unlist(claims_embeddings$embeddings),
  ncol = 1536, byrow = TRUE
)

dim(embeddings_mat)


# Similarity

embeddings_similarity <- embeddings_mat / sqrt(rowSums(embeddings_mat * embeddings_mat))
embeddings_similarity <- embeddings_similarity %*% t(embeddings_similarity)

dim(embeddings_similarity)


# Analyze example

claims |>
  slice(7) |>
  select(class, description)

enframe(embeddings_similarity[7, ], name = "claim", value = "similarity") |>
  arrange(-similarity)

claims |>
  slice(c(441, 498, 144)) |>
  select(CASO_ID, description, class)


# PCA

set.seed(1234)

claims_pca <- irlba::prcomp_irlba(embeddings_mat, n = 20)

augmented_pca <- as_tibble(claims_pca$x) |>
  bind_cols(claims)

augmented_pca |>
  ggplot(aes(PC1, PC2, color = class)) +
  geom_point(size = 3, alpha = 0.7) +
  theme_bw()

ggsave("plots/pca.png", dpi = 320, width = 12, height = 9)
