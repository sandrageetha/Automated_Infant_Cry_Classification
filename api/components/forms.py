from flask_wtf import FlaskForm
from wtforms import FileField
from wtforms.validators import DataRequired

class FileUploadForm(FlaskForm):
    file = FileField('File', validators=[DataRequired()])

    def file_is_valid(self):
        if not self.file.data:
            return False, ["File is required"]
        if "audio" not in self.file.data.content_type:
            return False, ["The file format is not supported. Please upload an audio file."]
        return True, []