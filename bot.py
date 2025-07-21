import os
import torch.nn as nn
import torch

from PIL import Image
from torchvision import models, transforms
from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.types.input_file import FSInputFile
from aiogram.filters import Command
import asyncio

from dotenv import load_dotenv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_model = models.densenet161()
best_model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(best_model.classifier.in_features, 1)
)
best_model.load_state_dict(torch.load('./models/densenet161_besttry.pth', map_location=device))
best_model.to(device)
best_model.eval()

def predict_cancer(image_path: str) -> float:
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.4875] * 3, [0.2456] * 3),
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)
    best_model.eval()

    with torch.no_grad():
        output = best_model(image_tensor)
        probability = torch.sigmoid(output).item()

    return probability

load_dotenv()
TOKEN = os.getenv("API_TOKEN")
bot = Bot(token=TOKEN)
dp = Dispatcher()

async def set_bot_commands():
    commands = [
        types.BotCommand(command="start", description="📜 Начать")
    ]
    await bot.set_my_commands(commands)

inline_kb = InlineKeyboardMarkup(
    inline_keyboard=[
        [InlineKeyboardButton(text="💡 Загрузить рентгеновский снимок", callback_data="load")],
        [InlineKeyboardButton(text="🔗 Шляпа какая-то", callback_data="demo")],
    ]
)

@dp.message(Command("start"))
async def start_command(message: types.Message):
    await message.answer("Выберите действие:", reply_markup=inline_kb)

@dp.callback_query()
async def handle_callback(query: types.CallbackQuery):
    if query.data == "load":
        await query.message.answer("Отправьте мне рентгеновский снимок в виде фото.")
    elif query.data == "demo":
        try:
            photo = FSInputFile("prod.jpg")
            await query.message.answer_photo(photo)
        except Exception as e:
            await query.message.answer(f"⚠️ Ошибка при отправке изображения: {e}")
    await query.answer()

# Обработчик загруженных изображений
@dp.message(lambda message: message.photo)
async def handle_photo(message: types.Message):
    photo = message.photo[-1]
    file_id = photo.file_id

    file_info = await bot.get_file(file_id)
    file_path = file_info.file_path
    destination = f"photo_{message.from_user.id}.jpeg"
    await bot.download_file(file_path, destination)

    await message.answer("Фото получено! Анализирую... 🔍")

    try:
        result = predict_cancer(destination)
        if result >= 0.5:
            await message.answer(f"⚠️ Обнаружены признаки рака.\nВероятность: {result:.2f}")
        else:
            await message.answer(f"✅ Снимок чист.\nВероятность: {result:.2f}")
    except Exception as e:
        await message.answer(f"Произошла ошибка при анализе: {e}")

    os.remove(destination)


async def main():
    await set_bot_commands()
    print("Бот запущен...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())