import {MarkdownReader, OpenAI, OpenAIEmbedding, VectorStoreIndex} from "llamaindex";
import {serviceContextFromDefaults, storageContextFromDefaults} from "llamaindex";

async function main() {
  const reader = new MarkdownReader()
  const documents = await reader.loadData("./dataset/cv.md")

  const model = new OpenAI({
    model: "gpt-3.5-turbo-16k",
    apiKey: process.env.OPENAI_API_KEY,
  })
  const embed = new OpenAIEmbedding()

  const service = serviceContextFromDefaults({llm: model, embedModel: embed})
  const storage = await storageContextFromDefaults({persistDir: "./storage"})

  await VectorStoreIndex.fromDocuments(documents, {
    serviceContext: service,
    storageContext: storage,
  });
}

main().then(r => console.log(r));
