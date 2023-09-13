document.addEventListener('DOMContentLoaded', () => {
    const form = document.querySelector('form');
    const outputDiv = document.querySelector('#output');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(form);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData,
            });
            const data = await response.json();
            if (data.success) {
                // Tampilkan gambar hasil di outputDiv
                const img = document.createElement('img');
                img.src = data.output_image_url;
                outputDiv.innerHTML = '';
                outputDiv.appendChild(img);
            } else {
                alert('Gagal memproses gambar.');
            }
        } catch (error) {
            console.error(error);
            alert('Terjadi kesalahan saat mengunggah gambar.');
        }
    });
});
