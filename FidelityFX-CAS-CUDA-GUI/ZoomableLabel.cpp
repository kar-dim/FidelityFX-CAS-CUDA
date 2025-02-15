#include <Qlabel>
#include <QPixmap>
#include <Qt>
#include <QtMinMax>
#include <QWheelEvent>
#include <QWidget>
#include <ZoomableLabel.h>

#define CLAMP(x) qBound(1.0, x, 4.0)

ZoomableLabel::ZoomableLabel(QWidget* parent) : QLabel(parent) 
{
    setAlignment(Qt::AlignCenter);
    setScaledContents(false);
}

void ZoomableLabel::setImage(const QPixmap& pixmap) 
{
    scaleFactor = 1.0;
    updateImage(pixmap);
}

void ZoomableLabel::updateImage(const QPixmap& pixmap)
{
    originalPixmap = pixmap;
    scaleImage();
}

void ZoomableLabel::wheelEvent(QWheelEvent* event) 
{
    if (originalPixmap.isNull())
        return;
    scaleFactor = CLAMP(event->angleDelta().y() > 0 ? scaleFactor * 1.1 : scaleFactor / 1.1);
    scaleImage();
}

void ZoomableLabel::scaleImage() 
{
    if (!originalPixmap.isNull())
        setPixmap(originalPixmap.scaled(originalPixmap.size() * scaleFactor, Qt::KeepAspectRatio, Qt::SmoothTransformation));
}