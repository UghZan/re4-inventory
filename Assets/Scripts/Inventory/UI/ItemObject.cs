using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;
using TMPro;
using System.Linq;

public class ItemObject : MonoBehaviour, IBeginDragHandler, IEndDragHandler, IDragHandler
{
    Vector2Int storedPos;
    ItemStack keptStack;

    RectTransform rectTransform;
    Transform lastSlotTransform;
    Canvas parentCanvas;
    Vector2 lastPosition, dragPosition;
    public InventoryZoneUI droppedInto;
    bool isDragged;
    int lastRotation, gridSize, gridSizeHalf;

    InventoryZoneUI uiParent;
    [SerializeField] Image stackImage;
    [SerializeField] TextMeshProUGUI stackAmount;

    public void Init(InventoryZoneUI ui)
    {
        rectTransform = GetComponent<RectTransform>();
        UpdateUIParentZone(ui);
        parentCanvas = GetComponentInParent<Canvas>();
    }

    void UpdateUIParentZone(InventoryZoneUI ui, bool update = false)
    {
        uiParent = ui;

        gridSize = ui.GetGridVisualSize();
        gridSizeHalf = gridSize / 2;
        if(update) UpdateObject();
    }

    public void SetItem(ItemStack newStack)
    {
        keptStack = newStack;
        storedPos = newStack.GetPositionInZone();
    }
    public ItemStack GetItem()
    {
        return keptStack;
    }
    public void UpdateObject()
    {
        stackImage.sprite = keptStack.item.itemIcon;
        stackImage.rectTransform.sizeDelta = new Vector2(keptStack.item.itemSize.x * gridSize, keptStack.item.itemSize.y * gridSize);

        stackAmount.gameObject.SetActive(keptStack.item.itemStackSize > 1);
        stackAmount.text = keptStack.GetStackAmount().ToString();
        transform.SetParent(uiParent.itemsZone.transform);

        UpdateVisual();
        rectTransform.localScale = Vector2.one; //to prevent weird UI scaling issues

        Vector2 pos = uiParent.GetGridOffset() + new Vector2(storedPos.x * gridSize + gridSizeHalf * keptStack.GetRotatedSize().x, -storedPos.y * gridSize - gridSizeHalf * keptStack.GetRotatedSize().y);
        rectTransform.anchoredPosition = pos;

    }

    void UpdateVisual()
    {
        rectTransform.sizeDelta = new Vector2(keptStack.GetRotatedSize().x * gridSize, keptStack.GetRotatedSize().y * gridSize);
        stackImage.rectTransform.rotation = Quaternion.AngleAxis(keptStack.GetRotation() * 90, Vector3.forward);
    }

    public void OnBeginDrag(PointerEventData eventData)
    {
        //Debug.Log("Begin Dragging");
        lastPosition = rectTransform.anchoredPosition;
        lastSlotTransform = rectTransform.parent;
        lastRotation = keptStack.GetRotation();

        uiParent.ClearSpace(keptStack.GetPositionInZone(), keptStack.GetRotatedSize());
        transform.SetParent(uiParent.uiManager.tempDragParent.transform, true);

        dragPosition = rectTransform.position - Input.mousePosition;

        GetComponent<Image>().raycastTarget = false;
        isDragged = true;
    }

    void ResetToLast()
    {
        transform.SetParent(lastSlotTransform, true);
        keptStack.SetRotation(lastRotation);
        rectTransform.anchoredPosition = lastPosition;
        UpdateVisual();
    }

    void PlaceStackInNewZone(InventoryZoneUI newZone, Vector2Int newPos)
    {
        if (newZone.AddItemInZoneAt(this, newPos))
        {
            UpdateUIParentZone(newZone);
            transform.SetParent(newZone.transform, true);
        }
        else
            Debug.LogError("Failed to place an item");
    }

    void MoveStack(Vector2Int newPos)
    {
        transform.SetParent(lastSlotTransform, true);
        keptStack.SetPositionInZone(newPos);
        uiParent.FillSpace(newPos, keptStack.GetRotatedSize());
    }

    public void OnEndDrag(PointerEventData eventData)
    {
        isDragged = false;

        if (droppedInto == null) //if we place it in a free zone
        {
            //Debug.Log("Dropped into empty space");
            ResetToLast();
            GetComponent<Image>().raycastTarget = true;
            return;
        }
        GetComponent<Image>().raycastTarget = true;

        if (droppedInto.GetFittingTypes().Contains(ItemSlot.ALL) || droppedInto.GetFittingTypes().Contains(keptStack.item.itemSlot))
        {
            Vector2Int newPos = TransformToGrid(droppedInto);
            //Debug.Log(newPos + " " + keptStack.GetRotatedSize());
            bool isFreeSpaceUnderneath = droppedInto.CheckFreeSpace(newPos, keptStack.GetRotatedSize());
            if (isFreeSpaceUnderneath)
            {
                if (droppedInto.StackBelongsToZone(keptStack))
                {
                    Debug.Log("Moved stack");
                    MoveStack(newPos);
                }
                else
                {
                    InventoryZoneUI prevZone = uiParent;
                    Debug.Log("Placed new stack");
                    uiParent.RemoveItemFromZone(this);
                    PlaceStackInNewZone(droppedInto, newPos);
                }
            }
            else
            {
                if(droppedInto.AddItemInZone(this))
                {
                    Debug.Log("Still found a place, let's go");
                    UpdateUIParentZone(droppedInto);
                    transform.SetParent(droppedInto.transform, true);
                }
                else
                    ResetToLast();
            }
        }
        else
        {
            Debug.Log("Wrong slot");
            ResetToLast();
        }

        uiParent.UpdateItemVisual();
        //Debug.Log(uiParent);
    }

    void Update()
    {
        if (isDragged)
        {
            if (Input.GetMouseButtonDown(1))
            {
                keptStack.Rotate();
                UpdateVisual();
            }
        }
    }

    public void OnDrag(PointerEventData eventData)
    {
        Vector2 pos = dragPosition + (Vector2)Input.mousePosition;
        transform.position = pos;
    }

    Vector2Int TransformToGrid(InventoryZoneUI relativeZone)
    {
        transform.SetParent(droppedInto.transform);
        Vector2 input = rectTransform.anchoredPosition;
        Vector2Int newVec = new Vector2Int(Mathf.Max(0, Mathf.RoundToInt((input.x - keptStack.GetRotatedSize().x * gridSizeHalf) / gridSize)),
                        Mathf.Max(0, Mathf.RoundToInt((-input.y - keptStack.GetRotatedSize().y * gridSizeHalf) / gridSize)));
        return newVec;
    }
}
