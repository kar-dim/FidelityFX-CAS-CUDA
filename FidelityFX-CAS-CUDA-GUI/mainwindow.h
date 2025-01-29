#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QAction>
#include <QImage>
#include <QLabel>
#include <QMainWindow>
#include <QSize>
#include <QSlider>
#include <QString>
#include <QVBoxLayout>
#include <QWidget>


class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void openImage();
    void saveImage();
    void sliderValueChanged();

private:
    const QString imageDialogFilterText { "Images (*.png *.jpg *.bmp *.webp *.tiff)" };
    void setupMenu();
    void setupSlider(QSlider *slider, QLabel *label, const int value) const;
    void setupImageView();
    void setupMainWidget();
    void addSliderLayout(QVBoxLayout *mainLayout, QSlider *slider, QLabel *label);
    void updateImageView(const QImage& image);

    QImage userImage, sharpenedImage;
    QSlider *sharpenStrength, *contrastAdaption;
    QLabel *imageView, *sharpenStrengthLabel, *contrastAdaptionLabel;
    void* casObj;
    QAction *openImageAction, *saveImageAction;
    const QSize targetImageSize;
    bool userImageHasAlpha;
};

#endif // MAINWINDOW_H
