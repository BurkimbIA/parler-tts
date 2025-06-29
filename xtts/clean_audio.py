"""
Audio Enhancement Script using Resemble AI
Denoises and enhances audio files in a directory using the Resemble Enhance library.
"""

import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm
from pathlib import Path
from resemble_enhance.enhancer.inference import denoise, enhance


class AudioEnhancer:
    """Audio enhancement class using Resemble AI enhancement models."""
    
    def __init__(self):
        """Initialize the AudioEnhancer with device detection."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
    
    def denoise_and_enhance(self, audio_path, solver="Midpoint", nfe=64, tau=0.7, denoising=True):
        """
        Denoise and enhance a single audio file.
        
        Args:
            audio_path (Path): Path to the input audio file
            solver (str): Solver method for enhancement ("Midpoint", "Euler", etc.)
            nfe (int): Number of function evaluations
            tau (float): Time parameter for enhancement
            denoising (bool): Whether to apply strong denoising (True) or enhancement (False)
        
        Returns:
            tuple: ((sample_rate, denoised_audio), (sample_rate, enhanced_audio))
        """
        solver = solver.lower()
        nfe = int(nfe)
        lambd = 0.9 if denoising else 0.1
        
        dwav, sr = torchaudio.load(audio_path)
        dwav = dwav.mean(dim=0)
        
        wav_denoised, new_sr = denoise(dwav, sr, self.device)
        
        wav_enhanced, new_sr = enhance(
            dwav, sr, self.device, 
            nfe=nfe, solver=solver, lambd=lambd, tau=tau
        )
        
        wav_denoised = wav_denoised.cpu().numpy()
        wav_enhanced = wav_enhanced.cpu().numpy()
        
        return (new_sr, wav_denoised), (new_sr, wav_enhanced)
    
    def process_directory(self, input_dir, output_dir, audio_format="wav", 
                         solver="Midpoint", nfe=64, tau=0.7, denoising=True,
                         use_enhanced=True):
        """
        Process all audio files in a directory.
        
        Args:
            input_dir (str): Input directory containing audio files
            output_dir (str): Output directory for processed files
            audio_format (str): Audio file format to process
            solver (str): Solver method for enhancement
            nfe (int): Number of function evaluations
            tau (float): Time parameter for enhancement
            denoising (bool): Whether to apply strong denoising
            use_enhanced (bool): Use enhanced version (True) or denoised only (False)
        
        Returns:
            int: Number of files successfully processed
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        audio_files = list(input_path.rglob(f'*.{audio_format}'))
        print(f"Found {len(audio_files)} {audio_format.upper()} files to process")
        
        if not audio_files:
            print("No audio files found!")
            return 0
        
        successful_count = 0
        
        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            try:
                denoised_output, enhanced_output = self.denoise_and_enhance(
                    audio_path=audio_file,
                    solver=solver,
                    nfe=nfe,
                    tau=tau,
                    denoising=denoising
                )
                
                sample_rate, audio_array = enhanced_output if use_enhanced else denoised_output
                
                output_filename = output_path / audio_file.name
                
                sf.write(
                    str(output_filename),
                    audio_array,
                    sample_rate,
                    format=audio_format.lower()
                )
                
                successful_count += 1
                
            except Exception as e:
                print(f"Error processing {audio_file.name}: {str(e)}")
                continue
        
        print(f"Successfully processed {successful_count}/{len(audio_files)} audio files")
        return successful_count
    
    def verify_output(self, output_dir, audio_format="wav"):
        """
        Verify the output directory and count processed files.
        
        Args:
            output_dir (str): Directory to verify
            audio_format (str): Audio file format to count
        
        Returns:
            int: Number of files found in output directory
        """
        output_path = Path(output_dir)
        if not output_path.exists():
            print(f"Output directory {output_dir} does not exist!")
            return 0
        
        processed_files = list(output_path.rglob(f'*.{audio_format}'))
        print(f"Found {len(processed_files)} processed {audio_format.upper()} files in output directory")
        return len(processed_files)


def main():
    """Main function to run the audio enhancement pipeline."""
    
    CONFIG = {
        'input_dir': './dataset/wavs',
        'output_dir': './dataset/wavs_enhanced',
        'audio_format': 'wav',
        'solver': 'Midpoint',
        'nfe': 64,
        'tau': 0.7,
        'denoising': True,
        'use_enhanced': True  # Use enhanced version instead of just denoised
    }
    
    enhancer = AudioEnhancer()
    
    successful_count = enhancer.process_directory(**CONFIG)
    
    if successful_count > 0:
        enhancer.verify_output(CONFIG['output_dir'], CONFIG['audio_format'])
    else:
        print("No files were successfully processed!")


if __name__ == "__main__":
    main()