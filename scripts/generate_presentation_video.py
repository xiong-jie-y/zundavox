from zundavox.presentation_video_generator import PresentationVideoGenerator
import click
from zundavox.config import get_cfg_defaults

@click.command()
@click.option('--script-path')
@click.option('--background-video-path')
@click.option('--pdf-slide-path')
@click.option('--character')
@click.option("--output-video-path")
@click.option("--output-subtitle-path")
@click.option("--config-file")
def main(script_path, output_video_path, output_subtitle_path, background_video_path, pdf_slide_path, character, config_file):
    config = get_cfg_defaults()
    if config_file is not None:
        config.merge_from_file(config_file)
    config.freeze()
    generator = PresentationVideoGenerator(background_video_path, pdf_slide_path, character, config)
    generator.generate_video(output_video_path, script_path, 1.0, output_subtitle_path, background_video_path, pdf_slide_path, character)

if __name__ == "__main__":
    main()