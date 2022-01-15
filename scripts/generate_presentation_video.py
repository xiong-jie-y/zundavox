from zundavox.presentation_video_generator import PresentationVideoGenerator
import click

@click.command()
@click.option('--script-path')
@click.option('--background-video-path')
@click.option('--pdf-slide-path')
@click.option('--character')
@click.option("--output-video-path")
@click.option("--output-subtitle-path")
def main(script_path, output_video_path, output_subtitle_path, background_video_path, pdf_slide_path, character):
    generator = PresentationVideoGenerator(background_video_path, pdf_slide_path, character)
    generator.generate_video(output_video_path, script_path, 1.0, output_subtitle_path, background_video_path, pdf_slide_path, character)

if __name__ == "__main__":
    main()