import {info, warning,getInput} from '@actions/core'
import {Pinecone} from '@pinecone-database/pinecone'
import {pipeline} from '@xenova/transformers'

export interface SOP {
  text: string
  id?: string
  score?: number
  metadata?: Record<string, any>
}

let embedder: any = null
let pineconeClient: Pinecone | null = null
let pineconeIndex: any = null

/**
 * Initialize the embedding pipeline (lazy initialization)
 */
async function initializeEmbedder(): Promise<void> {
  if (embedder === null) {
    try {
      info('Initializing embedding model: all-MiniLM-L6-v2')
      embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2')
      info('Embedding model initialized successfully')
    } catch (e: any) {
      warning(`Failed to initialize embedding model: ${e.message}`)
      throw e
    }
  }
}

/**
 * Initialize Pinecone client and index (lazy initialization)
 */
async function initializePinecone(): Promise<void> {
  if (pineconeClient === null) {
    const apiKey = getInput('pinecone_api_key')
    const host = getInput('pinecone_host')
    // Extract index name from host if provided, or use environment variable
    // Host format: sop-embeddings-2vib48a.svc.aped-4627-b74a.pinecone.io
    // Index name might be: sop-embeddings-2vib48a
    let indexName = getInput('pinecone_index')
    if (!indexName && host) {
      // Extract index name from host (part before .svc.)
      const match = host.match(/^([^.]+)\.svc\./)
      if (match) {
        indexName = match[1]
      }
    }
    if (!indexName) {
      indexName = 'sop-embeddings'
    }

    if (!apiKey) {
      throw new Error('PINECONE_API_KEY environment variable is not set')
    }

    try {
      info(`Initializing Pinecone client`)
      // Initialize Pinecone client with API key
      pineconeClient = new Pinecone({
        apiKey
      })

      info(`Connecting to Pinecone index: ${indexName}`)
      pineconeIndex = pineconeClient.index(indexName)
      info('Pinecone client initialized successfully')
    } catch (e: any) {
      warning(`Failed to initialize Pinecone client: ${e.message}`)
      throw e
    }
  }
}

/**
 * Embed text using the all-MiniLM-L6-v2 model
 * Returns a 384-dimensional vector
 */
export async function embedText(text: string): Promise<number[]> {
  try {
    await initializeEmbedder()
    if (embedder === null) {
      throw new Error('Embedder not initialized')
    }

    const output = await embedder(text, {
      pooling: 'mean',
      normalize: true
    })

    // Convert tensor to array - handle different output formats
    let vector: number[]
    if (output && typeof output.data !== 'undefined') {
      // If output has .data property (TypedArray)
      vector = Array.from(output.data) as number[]
    } else if (Array.isArray(output)) {
      // If output is already an array
      vector = output as number[]
    } else if (output && typeof output.tolist === 'function') {
      // If output has tolist method
      vector = output.tolist() as number[]
    } else {
      // Try to extract data from nested structure
      const data = (output as any).data || output
      vector = Array.from(data) as number[]
    }

    if (vector.length !== 384) {
      warning(
        `Expected 384-dimensional vector, got ${vector.length} dimensions`
      )
    }

    return vector
  } catch (e: any) {
    warning(`Failed to embed text: ${e.message}`)
    throw e
  }
}

/**
 * Get relevant SOPs for a diff chunk by querying Pinecone
 * Returns top 3 most relevant SOPs using cosine similarity
 */
export async function getRelevantSops(diffChunk: string): Promise<SOP[]> {
  try {
    // Initialize Pinecone if not already done
    await initializePinecone()

    if (pineconeIndex === null) {
      warning('Pinecone index not initialized, skipping SOP retrieval')
      return []
    }

    // Embed the diff chunk
    info(`Embedding diff chunk (${diffChunk.length} characters)`)
    const vector = await embedText(diffChunk)

    // Query Pinecone for top 3 relevant SOPs
    info('Querying Pinecone for relevant SOPs')
    const queryResponse = await pineconeIndex.query({
      vector,
      topK: 3,
      includeMetadata: true
    })

    if (!queryResponse.matches || queryResponse.matches.length === 0) {
      info('No relevant SOPs found for this diff chunk')
      return []
    }

    // Convert matches to SOP format
    const sops: SOP[] = queryResponse.matches.map(match => {
      const sop: SOP = {
        text: (match.metadata?.text as string) || '',
        id: match.id,
        score: match.score,
        metadata: match.metadata as Record<string, any>
      }
      return sop
    })

    info(
      `Retrieved ${sops.length} relevant SOP(s) for diff chunk (scores: ${sops.map(s => s.score?.toFixed(3)).join(', ')})`
    )

    return sops
  } catch (e: any) {
    warning(`Failed to retrieve relevant SOPs: ${e.message}`)
    // Return empty array on error to gracefully handle failures
    return []
  }
}

/**
 * Format SOPs into a readable string for inclusion in prompts
 */
export function formatSopsForPrompt(sops: SOP[]): string {
  if (sops.length === 0) {
    return ''
  }

  const sopTexts = sops
    .map((sop, index) => {
      return `### Relevant SOP ${index + 1}${sop.id ? ` (ID: ${sop.id})` : ''}${sop.score !== undefined ? ` (relevance: ${sop.score.toFixed(3)})` : ''}

${sop.text}`
    })
    .join('\n\n')

  return `
<relevant_sops>
${sopTexts}
</relevant_sops>
`
}

