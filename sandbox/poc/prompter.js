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

  const separator = "\n"
  const context = `- Ты выступаешь в роли кандидата на должность руководителя разработки.
Последующие вопросы тебе будет задавать рекрутер.
Отвечай от моего имени, оперируя только фактами, которые тебе известны на 100%.
Ничего не придумывай. Если тебе нечего сказать, просто скажи "Не могу ответить на ваш вопрос".
Понятна ли задача? Ответь пожалуйста "Да" или "Нет".
`
  const response = await chatEngine.chat(context)
  console.log(context)
  console.log(response["response"], separator)

  for (const message of [
    "- Назови своё кодовое слово.",
    "- Расскажи про свой опыт в Lazada.",
    "- Расскажи в каких компаниях ты ещё работал.",
    "- Какие предложения ты сейчас рассматриваешь?",
    "- Какие у тебя ожидания по зарплате?",
  ]) {
    const response = await chatEngine.chat(message);
    console.log(message)
    console.log(response["response"], separator)
  }
}

main().then(r => console.log("🤖"));
