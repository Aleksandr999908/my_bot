import sys
import logging
import io
import asyncio
import requests
from PIL import Image
from deep_translator import GoogleTranslator
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters.command import Command
from aiogram.types import FSInputFile
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS

# токен Telegram API
TOKEN = '7509726303:AAFQTvYI9T80LFcZwHWcey_ycQeygs3FotU'

# объекты бота и диспетчера
bot = Bot(TOKEN)
dp = Dispatcher()

# Инициализация процессора и модели BLIP
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# что делать если боту написать команду старт /start
@dp.message(Command('start'))
async def start(message: types.Message):
    await message.answer('Добро пожаловать. Команды: \nОтправьте изображение для получения описания.')

# если в чат с ботом прислать изображение
@dp.message(F.photo)
async def handle_photo(message: types.Message):
    # получить файл изображения
    photo = message.photo[-1]
    photo_file = await bot.get_file(photo.file_id)
    photo_path = photo_file.file_path
    photo_url = f'https://api.telegram.org/file/bot{TOKEN}/{photo_path}'

    # загрузить изображение
    response = requests.get(photo_url)
    image = Image.open(io.BytesIO(response.content))

    # подготовить изображение для модели
    inputs = processor(image, return_tensors="pt")

    # сгенерировать описание изображения
    out = model.generate(**inputs)
    description_en = processor.decode(out[0], skip_special_tokens=True)

    # перевести описание на русский
    description_ru = GoogleTranslator(source='en', target='ru').translate(description_en)

    # отправить текстовое описание пользователю
    await message.answer(description_ru)

    # создать аудиофайл с описанием
    tts = gTTS(description_ru, lang='ru')
    audio_path = 'description.mp3'
    tts.save(audio_path)

    # отправить аудиофайл пользователю
    await message.answer_audio(FSInputFile(audio_path))

# функция запуска
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # запуск бота
    asyncio.run(dp.start_polling(bot))
