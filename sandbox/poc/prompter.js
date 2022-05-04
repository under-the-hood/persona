import {ContextChatEngine, OpenAI, OpenAIEmbedding, VectorStoreIndex} from "llamaindex";
import {serviceContextFromDefaults, storageContextFromDefaults} from "llamaindex";

async function main() {
  const model = new OpenAI({
    model: "gpt-3.5-turbo-16k",
    apiKey: process.env.OPENAI_API_KEY,
  })
  const embed = new OpenAIEmbedding()

  const service = serviceContextFromDefaults({llm: model, embedModel: embed})
  const storage = await storageContextFromDefaults({persistDir: "./storage"});
  const index = await VectorStoreIndex.init({
    serviceContext: service,
    storageContext: storage,
  });

  const retriever = index.asRetriever();
  const chatEngine = new ContextChatEngine({retriever});

  for (const message of [
    "Расскажи вкратце про мой опыт работы в хронологическом порядке",
  ]) {
    const response = await chatEngine.chat(message);
    console.log(response["response"])
  }
}

main().then(r => console.log(r));
