import sys
import os
import torch
import torchaudio
import soundfile as sf
from datetime import datetime
import sounddevice as sd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTextEdit, QPushButton,
                             QVBoxLayout, QWidget, QLabel, QComboBox, QSpinBox,
                             QFileDialog, QHBoxLayout, QGroupBox, QCheckBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class TTSApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Генератор аудио из текста с ударениями (v3.1)")
        self.setGeometry(100, 100, 800, 600)

        # Инициализация модели
        self.device = torch.device('cpu')
        self.model = None
        self.temp_file = "temp_audio.wav"
        self.model_dir = "silero_models"
        self.model_file = "v3_1_ru.pt"

        # Основные элементы интерфейса
        self.init_ui()

        # Загрузка модели после инициализации UI
        self.load_model()

    def init_ui(self):
        """Инициализация пользовательского интерфейса"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Группа настроек
        settings_group = QGroupBox("Настройки генерации")
        settings_layout = QHBoxLayout()

        # Выбор голоса
        self.voice_combo = QComboBox()
        self.voice_combo.addItems([
            'aidar (мужской)',
            'baya (женский)',
            'kseniya (женский)',
            'xenia (женский)',
            'eugene (мужской)'
        ])
        self.voice_combo.setCurrentIndex(1)  # baya по умолчанию

        # Частота дискретизации
        self.sample_rate_spin = QSpinBox()
        self.sample_rate_spin.setRange(8000, 48000)
        self.sample_rate_spin.setValue(24000)
        self.sample_rate_spin.setSingleStep(8000)
        self.sample_rate_spin.setToolTip("Рекомендуется 24000 или 48000 для лучшего качества")

        # Битность
        self.bit_depth_combo = QComboBox()
        self.bit_depth_combo.addItems(['16 бит', '24 бит', '32 бит'])
        self.bit_depth_combo.setCurrentIndex(0)

        # Чекбокс для ударений
        self.stress_checkbox = QCheckBox("Режим ударений (через '+')")
        self.stress_checkbox.setChecked(True)
        self.stress_checkbox.setToolTip("Например: '+у+диви+тельно'")

        settings_layout.addWidget(QLabel("Голос:"))
        settings_layout.addWidget(self.voice_combo)
        settings_layout.addWidget(QLabel("Частота (Гц):"))
        settings_layout.addWidget(self.sample_rate_spin)
        settings_layout.addWidget(QLabel("Битность:"))
        settings_layout.addWidget(self.bit_depth_combo)
        settings_layout.addWidget(self.stress_checkbox)
        settings_group.setLayout(settings_layout)

        # Текстовое поле
        self.text_edit = QTextEdit()
        self.text_edit.setFont(QFont("Arial Unicode MS", 12))
        self.text_edit.setPlaceholderText(
            "Введите текст для преобразования...\nДля ударений используйте + перед гласной: '+у+диви+тельно'")
        self.text_edit.setPlainText(
            """Номер набран неправильно. Номер++а спецслужб необходимо набирать в трёх зн++ачном формате.
            Например:
            Служба спасения — сто один.
            Полиция — сто два.
            Ск++орая помощь — сто три.
            Газовая служба — сто четыре.
            Пожалуйста, Скоррект++ируйте  набор номера. +++"""
        )

        # Кнопки управления
        self.btn_generate = QPushButton("Сгенерировать аудио (Ctrl+G)")
        self.btn_play = QPushButton("Воспроизвести (Ctrl+P)")
        self.btn_save = QPushButton("Сохранить как... (Ctrl+S)")

        # Назначение горячих клавиш
        self.btn_generate.setShortcut("Ctrl+G")
        self.btn_play.setShortcut("Ctrl+P")
        self.btn_save.setShortcut("Ctrl+S")

        # Отключаем кнопки до загрузки модели
        self.btn_generate.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.btn_save.setEnabled(False)

        # Соединение сигналов
        self.btn_generate.clicked.connect(self.generate_audio)
        self.btn_play.clicked.connect(self.play_audio)
        self.btn_save.clicked.connect(self.save_audio)

        # Расположение кнопок
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_generate)
        btn_layout.addWidget(self.btn_play)
        btn_layout.addWidget(self.btn_save)

        # Сборка интерфейса
        main_layout.addWidget(settings_group)
        main_layout.addWidget(QLabel("Текст:"))
        main_layout.addWidget(self.text_edit)
        main_layout.addLayout(btn_layout)

    def load_model(self):
        """Загрузка модели Silero TTS v3_1_ru"""
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            model_path = os.path.join(self.model_dir, self.model_file)

            # Автозагрузка модели если отсутствует
            if not os.path.exists(model_path):
                self.statusBar().showMessage("Скачивание модели... Это может занять несколько минут")
                QApplication.processEvents()

                torch.hub.download_url_to_file(
                    'https://models.silero.ai/models/tts/ru/v3_1_ru.pt',
                    model_path
                )

            # Загрузка модели
            self.statusBar().showMessage("Загрузка модели...")
            QApplication.processEvents()

            self.model = torch.package.PackageImporter(model_path).load_pickle("tts_models", "model")
            self.model.to(self.device)

            # Активируем кнопки после успешной загрузки
            self.btn_generate.setEnabled(True)
            self.statusBar().showMessage("Модель успешно загружена", 3000)

        except Exception as e:
            self.statusBar().showMessage(f"Ошибка загрузки модели: {str(e)}", 10000)
            print(f"Ошибка загрузки модели: {str(e)}")

    def process_stresses(self, text):
        """Обработка ударений в тексте"""
        result = []
        i = 0
        while i < len(text):
            if text[i] == '+' and i + 1 < len(text) and text[i + 1].lower() in "аеёиоуыэюя":
                result.append(text[i + 1] + '\u0301')  # Добавляем ударение
                i += 2
            else:
                result.append(text[i])
                i += 1
        return ''.join(result)

    def get_voice_name(self):
        """Получение имени голоса из комбобокса"""
        return self.voice_combo.currentText().split()[0]

    def get_subtype(self):
        """Получение типа аудио по битности"""
        return {
            '16 бит': 'PCM_16',
            '24 бит': 'PCM_24',
            '32 бит': 'PCM_32'
        }[self.bit_depth_combo.currentText()]

    def generate_audio(self):
        """Генерация аудио из текста"""
        if self.model is None:
            self.statusBar().showMessage("Модель не загружена!", 5000)
            return

        text = self.text_edit.toPlainText()
        if not text.strip():
            self.statusBar().showMessage("Введите текст для преобразования!", 5000)
            return

        try:
            # Обработка ударений
            if self.stress_checkbox.isChecked():
                text = self.process_stresses(text)

            voice = self.get_voice_name()
            target_sr = self.sample_rate_spin.value()

            # Генерация аудио
            self.statusBar().showMessage("Генерация аудио...")
            QApplication.processEvents()

            audio = self.model.apply_tts(
                text=text,
                speaker=voice,
                sample_rate=48000  # Всегда генерируем в 48 кГц
            )

            # Ресемплинг до нужной частоты
            if target_sr != 48000:
                audio = torchaudio.functional.resample(audio, 48000, target_sr)

            # Сохранение во временный файл
            subtype = self.get_subtype()
            sf.write(
                self.temp_file,
                audio.numpy(),
                samplerate=target_sr,
                subtype=subtype,
                format='WAV'
            )

            # Активируем кнопки воспроизведения и сохранения
            self.btn_play.setEnabled(True)
            self.btn_save.setEnabled(True)

            self.statusBar().showMessage("✅ Аудио успешно сгенерировано!", 3000)

        except Exception as e:
            self.statusBar().showMessage(f"❌ Ошибка генерации: {str(e)}", 5000)
            print(f"Ошибка генерации: {str(e)}")

    def play_audio(self):
        """Воспроизведение сгенерированного аудио"""
        if not os.path.exists(self.temp_file):
            self.statusBar().showMessage("❌ Сначала сгенерируйте аудио!", 5000)
            return

        try:
            audio, sr = sf.read(self.temp_file)
            sd.play(audio, sr)
            sd.wait()
            self.statusBar().showMessage("🔊 Воспроизведение завершено", 3000)
        except Exception as e:
            self.statusBar().showMessage(f"❌ Ошибка воспроизведения: {str(e)}", 5000)
            print(f"Ошибка воспроизведения: {str(e)}")

    def save_audio(self):
        """Сохранение аудиофайла"""
        if not os.path.exists(self.temp_file):
            self.statusBar().showMessage("❌ Сначала сгенерируйте аудио!", 5000)
            return

        try:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            default_name = f"audio_{current_time}.wav"

            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Сохранить аудиофайл",
                default_name,
                "WAV files (*.wav);;All files (*)"
            )

            if file_path:
                # Добавляем расширение .wav если его нет
                if not file_path.lower().endswith('.wav'):
                    file_path += '.wav'

                with open(self.temp_file, 'rb') as src, open(file_path, 'wb') as dst:
                    dst.write(src.read())

                self.statusBar().showMessage(f"💾 Файл сохранен: {os.path.basename(file_path)}", 5000)

        except Exception as e:
            self.statusBar().showMessage(f"❌ Ошибка сохранения: {str(e)}", 5000)
            print(f"Ошибка сохранения: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = TTSApp()
    window.show()
    sys.exit(app.exec_())