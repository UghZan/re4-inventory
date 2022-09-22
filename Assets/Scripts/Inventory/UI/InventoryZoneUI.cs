using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;

public class InventoryZoneUI : MonoBehaviour, IDropHandler
{
    [Header("Settings")]
    [SerializeField] bool shouldFitSize;
    int maxHeight; //how much place does inventory takes vertically, needed for scrolling correctly
     private Vector2 size;
    [SerializeField] private Vector2 itemOffset; //in case you need to correct where items are placed inside the inv zone
    public Vector2 GetGridOffset() => itemOffset;
    [SerializeField] private int gridVisualSize = 64; //how much pixels should 1 grid slot occupy
    public int GetGridVisualSize() => gridVisualSize;
    public Vector2 GetSizeInCanvas() => size;

    [Header("Pooling")]
    [SerializeField] ObjectPool itemObjectsPool;
    List<GameObject> itemObjects = new List<GameObject>();

    [Header("References")]
    public InventoryUI uiManager;
    public GameObject itemsZone;
    public Canvas parentCanvas;
    [SerializeField] InventoryZone representedInvZone;
    public InventoryZone GetRepresentedInvZone() => representedInvZone;

    bool needToUpdateItemObjects = true;

    public void Init()
    {
        representedInvZone.InitZone();
        size = RectTransformUtility.PixelAdjustRect(itemsZone.GetComponent<RectTransform>(), parentCanvas).size;
        UpdateItemVisual();
    }

    public ItemSlot[] GetFittingTypes()
    {
        return representedInvZone.fittingTypes;
    }

    public bool StackBelongsToZone(ItemStack stack)
    {
        return stack.parentZone == representedInvZone;
    }

    void IDropHandler.OnDrop(PointerEventData eventData)
    {
        if (eventData.pointerDrag != null)
        {
            if (eventData.pointerDrag.TryGetComponent(out ItemObject obj))
            {
                obj.droppedInto = this;
            }
        }
    }
    public void UpdateItemVisual()
    {
        if (needToUpdateItemObjects)
        {
            //Debug.Log(name + " updated");
            List<ItemStack> _items = representedInvZone.GetItemList();
            //return all current present item visuals back to pool
            for (int i = 0; i < itemObjects.Count; i++)
            {
                itemObjects[i].SetActive(false);
                itemObjects[i].transform.SetParent(itemObjectsPool.transform);
            }
            itemObjects.Clear();
            for (int i = 0; i < _items.Count; i++)
            {
                GameObject _itemVisual = itemObjectsPool.GetPooledObject();
                ItemObject io = _itemVisual.GetComponent<ItemObject>();
                io.Init(this);
                io.SetItem(_items[i]);
                io.UpdateObject();
                _itemVisual.SetActive(true);
                itemObjects.Add(_itemVisual);
            }
        }

        if (shouldFitSize)
        {
            maxHeight = representedInvZone.GetGridHeight();
            UpdateInventoryZoneHeight();
        }
    }

    void UpdateInventoryZoneHeight()
    {
        itemsZone.GetComponent<RectTransform>().sizeDelta = new Vector2(itemsZone.GetComponent<RectTransform>().sizeDelta.x, (maxHeight + 1) * 32);
    }

    public bool CheckFreeSpace(Vector2Int gridPos, ItemGridMatrix matrix, int rotation)
    {
        return representedInvZone.CheckIfInvPositionIsFree(gridPos, matrix, rotation);
    }

    public void ClearSpace(Vector2Int gridPos, ItemGridMatrix matrix, int rotation)
    {
        representedInvZone.ClearOccupiedSpace(gridPos, matrix, rotation);
    }

    public void FillSpace(Vector2Int gridPos, ItemGridMatrix matrix, int rotation)
    {
        representedInvZone.FillOccupiedSpace(gridPos, matrix, rotation);
    }

    public bool RemoveItemFromZone(ItemObject visual)
    {
        bool result = representedInvZone.RemoveItem(visual.GetItem());
            needToUpdateItemObjects = true;
        return result;
    }
    public bool AddItemInZone(ItemObject visual)
    {
        bool result = representedInvZone.AddItem(visual.GetItem());
        if (result)
        {
            itemObjects.Add(visual.gameObject);
            needToUpdateItemObjects = true;
        }
        return result;
    }

    public bool AddItemInZoneAt(ItemObject visual, Vector2Int pos)
    {
        bool result = representedInvZone.AddItemAt(visual.GetItem(), pos);
        if (result)
        {
            itemObjects.Add(visual.gameObject);
            needToUpdateItemObjects = true;
        }
        return result;
    }
}
