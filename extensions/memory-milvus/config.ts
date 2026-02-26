export type MilvusMemoryConfig = {
  embedding: {
    provider: "openai";
    model?: string;
    apiKey: string;
    baseUrl?: string;
    dims?: number;
  };
  milvus: {
    address: string;
    token?: string;
    username?: string;
    password?: string;
    database?: string;
    collectionName?: string;
  };
  autoCapture?: boolean;
  autoRecall?: boolean;
  captureMaxChars?: number;
};

export const MEMORY_CATEGORIES = ["preference", "fact", "decision", "entity", "other"] as const;
export type MemoryCategory = (typeof MEMORY_CATEGORIES)[number];

const DEFAULT_MODEL = "text-embedding-3-small";
const DEFAULT_COLLECTION_NAME = "openclaw_memories";
export const DEFAULT_CAPTURE_MAX_CHARS = 500;

const EMBEDDING_DIMENSIONS: Record<string, number> = {
  "text-embedding-3-small": 1536,
  "text-embedding-3-large": 3072,
};

function assertAllowedKeys(value: Record<string, unknown>, allowed: string[], label: string) {
  const unknown = Object.keys(value).filter((key) => !allowed.includes(key));
  if (unknown.length === 0) {
    return;
  }
  throw new Error(`${label} has unknown keys: ${unknown.join(", ")}`);
}

export function vectorDimsForModel(model: string, explicitDims?: number): number {
  if (explicitDims && explicitDims > 0) {
    return explicitDims;
  }
  const dims = EMBEDDING_DIMENSIONS[model];
  if (!dims) {
    throw new Error(
      `Unknown embedding model: ${model}. Specify embedding.dims explicitly for custom models.`,
    );
  }
  return dims;
}

function resolveEnvVars(value: string): string {
  return value.replace(/\$\{([^}]+)\}/g, (_, envVar) => {
    const envValue = process.env[envVar];
    if (!envValue) {
      throw new Error(`Environment variable ${envVar} is not set`);
    }
    return envValue;
  });
}

function resolveEmbeddingModel(
  embedding: Record<string, unknown>,
  dims?: number,
): string {
  const model = typeof embedding.model === "string" ? embedding.model : DEFAULT_MODEL;
  // Validate known models; for unknown models, dims must be specified
  vectorDimsForModel(model, dims);
  return model;
}

export const milvusMemoryConfigSchema = {
  parse(value: unknown): MilvusMemoryConfig {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      throw new Error("memory config required");
    }
    const cfg = value as Record<string, unknown>;
    assertAllowedKeys(
      cfg,
      ["embedding", "milvus", "autoCapture", "autoRecall", "captureMaxChars"],
      "memory config",
    );

    const embedding = cfg.embedding as Record<string, unknown> | undefined;
    if (!embedding || typeof embedding.apiKey !== "string") {
      throw new Error("embedding.apiKey is required");
    }
    assertAllowedKeys(embedding, ["apiKey", "model", "baseUrl", "dims"], "embedding config");

    const milvus = cfg.milvus as Record<string, unknown> | undefined;
    if (!milvus || typeof milvus.address !== "string") {
      throw new Error("milvus.address is required");
    }
    assertAllowedKeys(
      milvus,
      ["address", "token", "username", "password", "database", "collectionName"],
      "milvus config",
    );

    const dims = typeof embedding.dims === "number" ? Math.floor(embedding.dims) : undefined;
    const model = resolveEmbeddingModel(embedding, dims);

    const captureMaxChars =
      typeof cfg.captureMaxChars === "number" ? Math.floor(cfg.captureMaxChars) : undefined;
    if (
      typeof captureMaxChars === "number" &&
      (captureMaxChars < 100 || captureMaxChars > 10_000)
    ) {
      throw new Error("captureMaxChars must be between 100 and 10000");
    }

    return {
      embedding: {
        provider: "openai",
        model,
        apiKey: resolveEnvVars(embedding.apiKey),
        baseUrl:
          typeof embedding.baseUrl === "string" ? resolveEnvVars(embedding.baseUrl) : undefined,
        dims,
      },
      milvus: {
        address: milvus.address as string,
        token: typeof milvus.token === "string" ? resolveEnvVars(milvus.token) : undefined,
        username: typeof milvus.username === "string" ? milvus.username : undefined,
        password:
          typeof milvus.password === "string" ? resolveEnvVars(milvus.password) : undefined,
        database: typeof milvus.database === "string" ? milvus.database : undefined,
        collectionName:
          typeof milvus.collectionName === "string"
            ? milvus.collectionName
            : DEFAULT_COLLECTION_NAME,
      },
      autoCapture: cfg.autoCapture === true,
      autoRecall: cfg.autoRecall !== false,
      captureMaxChars: captureMaxChars ?? DEFAULT_CAPTURE_MAX_CHARS,
    };
  },
  uiHints: {
    "embedding.apiKey": {
      label: "OpenAI API Key",
      sensitive: true,
      placeholder: "sk-proj-...",
      help: "API key for OpenAI embeddings (or use ${OPENAI_API_KEY})",
    },
    "embedding.model": {
      label: "Embedding Model",
      placeholder: DEFAULT_MODEL,
      help: "OpenAI embedding model to use",
    },
    "milvus.address": {
      label: "Milvus Address",
      placeholder: "localhost:19530",
      help: "Milvus gRPC endpoint address",
    },
    "milvus.token": {
      label: "Milvus Token",
      sensitive: true,
      help: "Token for token-based authentication (optional)",
    },
    "milvus.username": {
      label: "Milvus Username",
      help: "Username for credential-based authentication (optional)",
    },
    "milvus.password": {
      label: "Milvus Password",
      sensitive: true,
      help: "Password for credential-based authentication (optional)",
    },
    "milvus.database": {
      label: "Milvus Database",
      placeholder: "default",
      advanced: true,
    },
    "milvus.collectionName": {
      label: "Collection Name",
      placeholder: DEFAULT_COLLECTION_NAME,
      advanced: true,
    },
    autoCapture: {
      label: "Auto-Capture",
      help: "Automatically capture important information from conversations",
    },
    autoRecall: {
      label: "Auto-Recall",
      help: "Automatically inject relevant memories into context",
    },
    captureMaxChars: {
      label: "Capture Max Chars",
      help: "Maximum message length eligible for auto-capture",
      advanced: true,
      placeholder: String(DEFAULT_CAPTURE_MAX_CHARS),
    },
  },
};
